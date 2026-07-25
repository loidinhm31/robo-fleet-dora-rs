use crate::{CoordinatorConfig, ProfileCatalog};
use robo_rover_lib::{PowerProfile, ResourceSnapshot};

pub(crate) fn fresh_low_cpu(
    snapshot: Option<&ResourceSnapshot>,
    catalog: &ProfileCatalog,
    config: &CoordinatorConfig,
    profile: PowerProfile,
    now_wall_ms: u64,
) -> bool {
    let Some(snapshot) = snapshot else {
        return false;
    };
    if snapshot.sampled_at_ms < 0
        || snapshot.sampled_at_ms as u64 > now_wall_ms
        || now_wall_ms.saturating_sub(snapshot.sampled_at_ms as u64) > config.resource_freshness_ms
    {
        return false;
    }
    catalog.targets(profile).iter().all(|domain| {
        snapshot.domains.get(*domain).is_some_and(|usage| {
            usage
                .cpu_usage_percent
                .is_some_and(|cpu| cpu <= config.max_domain_cpu_percent)
                && usage.configured_node_count > 0
        })
    })
}
