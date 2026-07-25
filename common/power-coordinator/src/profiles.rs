use robo_rover_lib::{LifecycleRole, PowerProfile};
use std::collections::{BTreeMap, BTreeSet};

const FORBIDDEN_TARGETS: [&str; 8] = [
    "rover-controller",
    "arm-controller",
    "watchdog",
    "zenoh-bridge",
    "orchestra-bridge",
    "resource-monitor",
    "recording-scheduler",
    "power-coordinator",
];

#[derive(Debug, Clone)]
pub struct ProfileCatalog {
    targets: BTreeMap<PowerProfileKey, Vec<&'static str>>,
    dependencies: Vec<(&'static str, &'static str)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum PowerProfileKey {
    Dormant,
    IdleListening,
    ScheduledCapture,
    NormalRover,
    OrchestraSpeech,
}

impl From<PowerProfile> for PowerProfileKey {
    fn from(value: PowerProfile) -> Self {
        match value {
            PowerProfile::Dormant => Self::Dormant,
            PowerProfile::IdleListening => Self::IdleListening,
            PowerProfile::ScheduledCapture => Self::ScheduledCapture,
            PowerProfile::NormalRover => Self::NormalRover,
            PowerProfile::OrchestraSpeech => Self::OrchestraSpeech,
        }
    }
}

impl ProfileCatalog {
    pub fn for_role(role: LifecycleRole) -> Result<Self, String> {
        let (targets, dependencies) = match role {
            LifecycleRole::Rover => (
                BTreeMap::from([
                    (PowerProfileKey::Dormant, vec![]),
                    (
                        PowerProfileKey::IdleListening,
                        vec!["audio-capture", "edge-voice"],
                    ),
                    (
                        PowerProfileKey::ScheduledCapture,
                        vec!["audio-capture", "gst-camera", "audio-playback"],
                    ),
                    (
                        PowerProfileKey::NormalRover,
                        vec![
                            "audio-capture",
                            "edge-voice",
                            "gst-camera",
                            "audio-playback",
                        ],
                    ),
                ]),
                vec![
                    ("audio-capture", "edge-voice"),
                    ("audio-capture", "audio-playback"),
                ],
            ),
            LifecycleRole::Orchestra => (
                BTreeMap::from([(
                    PowerProfileKey::OrchestraSpeech,
                    vec!["central-speech-recognizer"],
                )]),
                vec![],
            ),
        };
        let catalog = Self {
            targets,
            dependencies,
        };
        catalog.validate()?;
        Ok(catalog)
    }

    pub fn targets(&self, profile: PowerProfile) -> &[&'static str] {
        self.targets
            .get(&profile.into())
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    pub fn transition_targets(&self, from: PowerProfile, to: PowerProfile) -> Vec<&'static str> {
        let source: BTreeSet<_> = self.targets(from).iter().copied().collect();
        let destination: BTreeSet<_> = self.targets(to).iter().copied().collect();
        if self.rank(to) >= self.rank(from) {
            self.ordered()
                .into_iter()
                .filter(|item| destination.contains(item) && !source.contains(item))
                .collect()
        } else {
            let mut items: Vec<_> = self
                .ordered()
                .into_iter()
                .filter(|item| source.contains(item) && !destination.contains(item))
                .collect();
            items.reverse();
            items
        }
    }

    pub fn rank(&self, profile: PowerProfile) -> u8 {
        match profile {
            PowerProfile::Dormant => 0,
            PowerProfile::IdleListening => 1,
            PowerProfile::ScheduledCapture => 2,
            PowerProfile::NormalRover | PowerProfile::OrchestraSpeech => 3,
        }
    }

    fn ordered(&self) -> Vec<&'static str> {
        let mut seen = BTreeSet::new();
        self.targets
            .values()
            .flat_map(|items| items.iter().copied())
            .filter(|item| seen.insert(*item))
            .collect()
    }

    fn validate(&self) -> Result<(), String> {
        let all = self.ordered();
        if all.iter().any(|target| FORBIDDEN_TARGETS.contains(target)) {
            return Err("profile contains an always-on safety target".into());
        }
        if self
            .targets
            .values()
            .any(|items| items.iter().collect::<BTreeSet<_>>().len() != items.len())
        {
            return Err("profile contains duplicate targets".into());
        }
        if self
            .dependencies
            .iter()
            .any(|(before, after)| !all.contains(before) || !all.contains(after))
        {
            return Err("profile dependency references an unknown target".into());
        }
        let mut edges: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
        for (before, after) in &self.dependencies {
            edges.entry(before).or_default().push(after);
        }
        let mut visiting = BTreeSet::new();
        let mut visited = BTreeSet::new();
        fn visit<'a>(
            node: &'a str,
            edges: &BTreeMap<&'a str, Vec<&'a str>>,
            visiting: &mut BTreeSet<&'a str>,
            visited: &mut BTreeSet<&'a str>,
        ) -> bool {
            if !visiting.insert(node) {
                return false;
            }
            let acyclic = edges
                .get(node)
                .into_iter()
                .flatten()
                .all(|next| visited.contains(next) || visit(next, edges, visiting, visited));
            visiting.remove(node);
            visited.insert(node);
            acyclic
        }
        all.iter()
            .all(|target| {
                visited.contains(target) || visit(target, &edges, &mut visiting, &mut visited)
            })
            .then_some(())
            .ok_or_else(|| "profile dependencies are cyclic".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rover_profiles_keep_the_control_spine_out_of_lifecycle_targets() {
        let catalog = ProfileCatalog::for_role(LifecycleRole::Rover).unwrap();
        assert!(!catalog
            .targets(PowerProfile::NormalRover)
            .contains(&"rover-controller"));
    }
    #[test]
    fn sleep_closes_dependents_before_prerequisites() {
        let catalog = ProfileCatalog::for_role(LifecycleRole::Rover).unwrap();
        assert_eq!(
            catalog
                .transition_targets(PowerProfile::NormalRover, PowerProfile::Dormant)
                .last(),
            Some(&"audio-capture")
        );
    }
}
