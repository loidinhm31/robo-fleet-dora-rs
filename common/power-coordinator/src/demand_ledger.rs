use robo_rover_lib::{
    PowerDemand, PowerDemandAction, PowerReasonCode, PowerReservation, MAX_POWER_DEMANDS,
};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LedgerOutcome {
    Applied,
    Idempotent,
}

#[derive(Debug, Clone)]
struct DemandEntry {
    demand: PowerDemand,
    active: bool,
}

#[derive(Debug, Default, Clone)]
pub struct DemandLedger {
    demands: BTreeMap<(String, String), DemandEntry>,
    reservations: BTreeMap<String, PowerReservation>,
}

impl DemandLedger {
    pub fn apply(
        &mut self,
        demand: PowerDemand,
        now_ms: u64,
    ) -> Result<LedgerOutcome, PowerReasonCode> {
        if demand.expires_at_ms <= now_ms {
            return Err(PowerReasonCode::Expired);
        }
        let key = (demand.entity_id.clone(), demand.demand_id.clone());
        if let Some(entry) = self.demands.get_mut(&key) {
            if !entry.demand.same_immutable_payload(&demand) {
                return Err(PowerReasonCode::DuplicateMismatch);
            }
            if demand.renew_sequence < entry.demand.renew_sequence {
                return Ok(LedgerOutcome::Idempotent);
            }
            if demand.renew_sequence == entry.demand.renew_sequence {
                return (entry.demand == demand)
                    .then_some(LedgerOutcome::Idempotent)
                    .ok_or(PowerReasonCode::DuplicateMismatch);
            }
            if !entry.active || demand.action != PowerDemandAction::Renew {
                return Err(PowerReasonCode::Conflict);
            }
            entry.demand = demand;
            return Ok(LedgerOutcome::Applied);
        }
        if demand.action != PowerDemandAction::Acquire {
            return Err(PowerReasonCode::Conflict);
        }
        if self.demands.len() >= MAX_POWER_DEMANDS {
            return Err(PowerReasonCode::CapacityExceeded);
        }
        self.demands.insert(
            key,
            DemandEntry {
                demand,
                active: true,
            },
        );
        Ok(LedgerOutcome::Applied)
    }

    pub fn release(
        &mut self,
        entity_id: &str,
        demand_id: &str,
    ) -> Result<LedgerOutcome, PowerReasonCode> {
        let Some(entry) = self.demands.get_mut(&(entity_id.into(), demand_id.into())) else {
            return Err(PowerReasonCode::Conflict);
        };
        if !entry.active {
            return Ok(LedgerOutcome::Idempotent);
        }
        entry.active = false;
        Ok(LedgerOutcome::Applied)
    }

    pub fn register_reservation(
        &mut self,
        reservation: PowerReservation,
        now_ms: u64,
    ) -> Result<LedgerOutcome, PowerReasonCode> {
        if reservation.expires_at_ms <= now_ms {
            return Err(PowerReasonCode::Expired);
        }
        match self.reservations.get(&reservation.reservation_id) {
            Some(current) if current == &reservation => Ok(LedgerOutcome::Idempotent),
            Some(_) => Err(PowerReasonCode::DuplicateMismatch),
            None if self.reservations.len() >= MAX_POWER_DEMANDS => {
                Err(PowerReasonCode::CapacityExceeded)
            }
            None => {
                self.reservations
                    .insert(reservation.reservation_id.clone(), reservation);
                Ok(LedgerOutcome::Applied)
            }
        }
    }

    pub fn release_reservation(
        &mut self,
        reservation_id: &str,
    ) -> Result<LedgerOutcome, PowerReasonCode> {
        self.reservations
            .remove(reservation_id)
            .map(|_| LedgerOutcome::Applied)
            .ok_or(PowerReasonCode::Conflict)
    }

    pub fn active_demands(&mut self, now_ms: u64) -> impl Iterator<Item = &PowerDemand> {
        self.demands.values_mut().for_each(|entry| {
            if entry.demand.expires_at_ms <= now_ms {
                entry.active = false;
            }
        });
        self.demands
            .values()
            .filter(move |entry| {
                entry.active
                    && entry.demand.not_before_ms <= now_ms
                    && now_ms < entry.demand.expires_at_ms
            })
            .map(|entry| &entry.demand)
    }

    pub fn active_reservations(&self, now_ms: u64) -> impl Iterator<Item = &PowerReservation> {
        self.reservations
            .values()
            .filter(move |item| item.not_before_ms <= now_ms && now_ms < item.expires_at_ms)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use robo_rover_lib::{
        LifecycleRole, PowerAuthority, PowerDemandPriority, PowerDemandSource, PowerProfile,
        POWER_PROTOCOL_VERSION,
    };
    fn demand(action: PowerDemandAction, renew: u64, expires: u64) -> PowerDemand {
        PowerDemand {
            protocol_version: POWER_PROTOCOL_VERSION,
            demand_id: "f4f3e2d1-c0b9-48a7-9615-141312111000".into(),
            action,
            source: PowerDemandSource::Ui,
            priority: PowerDemandPriority::Normal,
            role: LifecycleRole::Rover,
            entity_id: "rover".into(),
            required_profile: PowerProfile::NormalRover,
            authority: PowerAuthority {
                epoch: 1,
                sequence: 1,
            },
            issued_at_ms: 1,
            not_before_ms: 1,
            expires_at_ms: expires,
            renew_sequence: renew,
        }
    }
    #[test]
    fn duplicate_and_reordered_renewals_never_extend_ttl() {
        let mut ledger = DemandLedger::default();
        ledger
            .apply(demand(PowerDemandAction::Acquire, 1, 10), 2)
            .unwrap();
        ledger
            .apply(demand(PowerDemandAction::Renew, 2, 20), 2)
            .unwrap();
        assert_eq!(
            ledger.apply(demand(PowerDemandAction::Renew, 1, 99), 2),
            Ok(LedgerOutcome::Idempotent)
        );
        assert_eq!(ledger.active_demands(21).count(), 0);
    }
    #[test]
    fn release_tombstone_cannot_be_reacquired() {
        let mut ledger = DemandLedger::default();
        ledger
            .apply(demand(PowerDemandAction::Acquire, 1, 10), 2)
            .unwrap();
        ledger
            .release("rover", "f4f3e2d1-c0b9-48a7-9615-141312111000")
            .unwrap();
        assert_eq!(
            ledger.apply(demand(PowerDemandAction::Acquire, 2, 20), 2),
            Err(PowerReasonCode::Conflict)
        );
    }
}
