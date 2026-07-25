use crate::ProfileCatalog;
use robo_rover_lib::{LifecycleDesiredState, LifecycleRole, LifecycleTarget, PowerProfile};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransitionDirection {
    Wake,
    Quiesce,
}

#[derive(Debug, Clone)]
pub struct TransitionPlan {
    pub stages: Vec<TransitionStage>,
}

#[derive(Debug, Clone)]
pub struct TransitionStage {
    pub desired_state: LifecycleDesiredState,
    pub nodes: Vec<String>,
}

pub fn plan_transition(
    catalog: &ProfileCatalog,
    from: PowerProfile,
    to: PowerProfile,
) -> TransitionPlan {
    let departing = catalog
        .transition_targets(from, PowerProfile::Dormant)
        .into_iter()
        .filter(|node| !catalog.targets(to).contains(node))
        .map(|node_id| TransitionStage {
            desired_state: LifecycleDesiredState::Quiesced,
            nodes: vec![node_id.into()],
        });
    let arriving = catalog
        .transition_targets(PowerProfile::Dormant, to)
        .into_iter()
        .filter(|node| !catalog.targets(from).contains(node))
        .map(|node_id| TransitionStage {
            desired_state: LifecycleDesiredState::Running,
            nodes: vec![node_id.into()],
        });
    TransitionPlan {
        stages: departing.chain(arriving).collect(),
    }
}

pub fn desired_state(direction: TransitionDirection) -> LifecycleDesiredState {
    match direction {
        TransitionDirection::Wake => LifecycleDesiredState::Running,
        TransitionDirection::Quiesce => LifecycleDesiredState::Quiesced,
    }
}

pub fn target(role: LifecycleRole, entity_id: &str, node_id: &str) -> LifecycleTarget {
    LifecycleTarget {
        role,
        entity_id: entity_id.into(),
        node_id: node_id.into(),
    }
}
