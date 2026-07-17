use robo_rover_lib::{TargetedMediaControl, RECORDING_PROTOCOL_VERSION};
use std::collections::BTreeSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MediaResource {
    Camera,
    Jpeg,
    Microphone,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct MediaDemand {
    entity_id: String,
    consumer_id: String,
    resource: MediaResource,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MediaDemandTransition {
    pub entity_id: String,
    pub resource: MediaResource,
    pub enabled: bool,
}

impl MediaDemandTransition {
    pub fn targeted_control(&self) -> TargetedMediaControl {
        TargetedMediaControl {
            protocol_version: RECORDING_PROTOCOL_VERSION,
            entity_id: self.entity_id.clone(),
            camera_enabled: (self.resource == MediaResource::Camera).then_some(self.enabled),
            jpeg_enabled: (self.resource == MediaResource::Jpeg).then_some(self.enabled),
            microphone_enabled: (self.resource == MediaResource::Microphone)
                .then_some(self.enabled),
        }
    }
}

/// In-memory effective-media state. One consumer can hold each resource once.
#[derive(Debug, Default)]
pub struct MediaDemandRegistry {
    demands: BTreeSet<MediaDemand>,
}

impl MediaDemandRegistry {
    pub fn acquire(
        &mut self,
        entity_id: impl Into<String>,
        consumer_id: impl Into<String>,
        resource: MediaResource,
    ) -> Option<MediaDemandTransition> {
        let demand = MediaDemand {
            entity_id: entity_id.into(),
            consumer_id: consumer_id.into(),
            resource,
        };
        // Target changes must use move_consumer_prefix. An ordinary duplicate
        // acquire never retargets a pinned consumer.
        if self.demands.iter().any(|existing| {
            existing.consumer_id == demand.consumer_id && existing.resource == demand.resource
        }) {
            return None;
        }
        let was_empty = !self.has_demand(&demand.entity_id, resource);
        self.demands
            .insert(demand.clone())
            .then(|| MediaDemandTransition {
                entity_id: demand.entity_id,
                resource,
                enabled: was_empty,
            })
            .filter(|transition| transition.enabled)
    }

    pub fn release(
        &mut self,
        entity_id: &str,
        consumer_id: &str,
        resource: MediaResource,
    ) -> Option<MediaDemandTransition> {
        let demand = MediaDemand {
            entity_id: entity_id.into(),
            consumer_id: consumer_id.into(),
            resource,
        };
        self.demands
            .remove(&demand)
            .then(|| MediaDemandTransition {
                entity_id: entity_id.into(),
                resource,
                enabled: false,
            })
            .filter(|_| !self.has_demand(entity_id, resource))
    }

    pub fn release_consumer(&mut self, consumer_id: &str) -> Vec<MediaDemandTransition> {
        let demands: Vec<_> = self
            .demands
            .iter()
            .filter(|demand| demand.consumer_id == consumer_id)
            .cloned()
            .collect();
        demands
            .into_iter()
            .filter_map(|demand| {
                self.release(&demand.entity_id, &demand.consumer_id, demand.resource)
            })
            .collect()
    }

    pub fn release_consumer_prefix(&mut self, prefix: &str) -> Vec<MediaDemandTransition> {
        let consumers: BTreeSet<_> = self
            .demands
            .iter()
            .filter(|demand| demand.consumer_id.starts_with(prefix))
            .map(|demand| demand.consumer_id.clone())
            .collect();
        consumers
            .into_iter()
            .flat_map(|consumer| self.release_consumer(&consumer))
            .collect()
    }

    /// Release a pinned resource without consulting mutable fleet selection.
    pub fn release_consumer_resource(
        &mut self,
        consumer_id: &str,
        resource: MediaResource,
    ) -> Option<MediaDemandTransition> {
        let demand = self
            .demands
            .iter()
            .find(|demand| demand.consumer_id == consumer_id && demand.resource == resource)
            .cloned()?;
        self.release(&demand.entity_id, consumer_id, resource)
    }

    /// Selection migration touches only this consumer; recorder demands remain pinned.
    pub fn move_consumer(
        &mut self,
        consumer_id: &str,
        entity_id: &str,
    ) -> Vec<MediaDemandTransition> {
        let resources: Vec<_> = self
            .demands
            .iter()
            .filter(|demand| demand.consumer_id == consumer_id)
            .cloned()
            .collect();
        let mut transitions = Vec::new();
        for demand in resources {
            if demand.entity_id != entity_id {
                if let Some(transition) =
                    self.release(&demand.entity_id, consumer_id, demand.resource)
                {
                    transitions.push(transition);
                }
                if let Some(transition) = self.acquire(entity_id, consumer_id, demand.resource) {
                    transitions.push(transition);
                }
            }
        }
        transitions
    }

    pub fn move_consumer_prefix(
        &mut self,
        prefix: &str,
        entity_id: &str,
    ) -> Vec<MediaDemandTransition> {
        let consumers: BTreeSet<_> = self
            .demands
            .iter()
            .filter(|demand| demand.consumer_id.starts_with(prefix))
            .map(|demand| demand.consumer_id.clone())
            .collect();
        consumers
            .into_iter()
            .flat_map(|consumer| self.move_consumer(&consumer, entity_id))
            .collect()
    }

    pub fn shutdown(&mut self) -> Vec<MediaDemandTransition> {
        let consumers: BTreeSet<_> = self
            .demands
            .iter()
            .map(|demand| demand.consumer_id.clone())
            .collect();
        consumers
            .into_iter()
            .flat_map(|consumer| self.release_consumer(&consumer))
            .collect()
    }

    fn has_demand(&self, entity_id: &str, resource: MediaResource) -> bool {
        self.demands
            .iter()
            .any(|demand| demand.entity_id == entity_id && demand.resource == resource)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn concurrent_entities_and_duplicate_events_are_independent() {
        let mut registry = MediaDemandRegistry::default();
        assert!(registry
            .acquire("rover-a", "recording:a", MediaResource::Jpeg)
            .is_some());
        assert!(registry
            .acquire("rover-b", "recording:b", MediaResource::Jpeg)
            .is_some());
        assert!(registry
            .acquire("rover-a", "recording:a", MediaResource::Jpeg)
            .is_none());
        assert!(registry
            .release("rover-a", "recording:a", MediaResource::Jpeg)
            .is_some());
        assert!(registry
            .release("rover-a", "recording:a", MediaResource::Jpeg)
            .is_none());
    }

    #[test]
    fn a_consumer_cannot_stop_another_consumers_resource() {
        let mut registry = MediaDemandRegistry::default();
        registry.acquire("rover-a", "browser:one", MediaResource::Camera);
        registry.acquire("rover-a", "recording:one", MediaResource::Camera);
        assert!(registry
            .release("rover-a", "browser:one", MediaResource::Camera)
            .is_none());
        assert_eq!(
            registry
                .release("rover-a", "recording:one", MediaResource::Camera)
                .unwrap()
                .enabled,
            false
        );
    }

    #[test]
    fn selection_migration_keeps_recorder_pinned() {
        let mut registry = MediaDemandRegistry::default();
        registry.acquire("rover-a", "browser:one", MediaResource::Camera);
        registry.acquire("rover-a", "recording:one", MediaResource::Camera);
        let changes = registry.move_consumer("browser:one", "rover-b");
        assert_eq!(
            changes,
            vec![MediaDemandTransition {
                entity_id: "rover-b".into(),
                resource: MediaResource::Camera,
                enabled: true
            }]
        );
        assert!(registry
            .release("rover-a", "recording:one", MediaResource::Camera)
            .is_some());
    }

    #[test]
    fn stop_releases_the_browsers_pinned_entity_after_another_selection_change() {
        let mut registry = MediaDemandRegistry::default();
        registry.acquire("rover-a", "browser:one", MediaResource::Jpeg);
        // The mutable UI selection is now rover-b, but browser:one never migrated.
        let stop = registry
            .release_consumer_resource("browser:one", MediaResource::Jpeg)
            .unwrap();
        assert_eq!(stop.entity_id, "rover-a");
        assert!(!stop.enabled);
    }

    #[test]
    fn duplicate_enable_cannot_retarget_a_pinned_consumer() {
        let mut registry = MediaDemandRegistry::default();
        registry.acquire("rover-a", "browser:one:stream", MediaResource::Jpeg);
        assert!(registry
            .acquire("rover-b", "browser:one:stream", MediaResource::Jpeg)
            .is_none());
        let stop = registry
            .release_consumer_resource("browser:one:stream", MediaResource::Jpeg)
            .unwrap();
        assert_eq!(stop.entity_id, "rover-a");
    }

    #[test]
    fn independent_browser_intents_cannot_release_each_other() {
        let mut registry = MediaDemandRegistry::default();
        registry.acquire("rover-a", "browser:one:stream", MediaResource::Camera);
        registry.acquire("rover-a", "browser:one:camera", MediaResource::Camera);
        assert!(registry
            .release_consumer_resource("browser:one:camera", MediaResource::Camera)
            .is_none());
        assert!(registry
            .release_consumer_resource("browser:one:stream", MediaResource::Camera)
            .is_some());
    }

    #[test]
    fn disconnect_rejection_and_shutdown_release_idempotently() {
        let mut registry = MediaDemandRegistry::default();
        registry.acquire("rover-a", "browser:one", MediaResource::Jpeg);
        registry.acquire("rover-b", "recording:one", MediaResource::Microphone);
        assert_eq!(registry.release_consumer("recording:missing"), Vec::new());
        assert_eq!(registry.release_consumer("browser:one").len(), 1);
        let shutdown = registry.shutdown();
        assert_eq!(shutdown.len(), 1);
        assert!(registry.shutdown().is_empty());
    }

    #[test]
    fn transition_serializes_only_the_changed_resource() {
        let transition = MediaDemandTransition {
            entity_id: "rover-a".into(),
            resource: MediaResource::Microphone,
            enabled: true,
        };
        let control = transition.targeted_control();
        assert_eq!(control.camera_enabled, None);
        assert_eq!(control.jpeg_enabled, None);
        assert_eq!(control.microphone_enabled, Some(true));
        control.validate().unwrap();
    }
}
