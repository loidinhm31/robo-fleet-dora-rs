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

#[derive(Debug, Clone)]
struct ReservationTombstone {
    reservation: PowerReservation,
    expires_at_ms: u64,
}

#[derive(Debug, Default, Clone)]
pub struct DemandLedger {
    demands: BTreeMap<(String, String), DemandEntry>,
    reservations: BTreeMap<String, PowerReservation>,
    reservation_tombstones: BTreeMap<String, ReservationTombstone>,
}

impl DemandLedger {
    pub fn apply(
        &mut self,
        demand: PowerDemand,
        now_ms: u64,
    ) -> Result<LedgerOutcome, PowerReasonCode> {
        self.prune_expired_demands(now_ms);
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
        if self.demands.values().filter(|entry| entry.active).count() >= MAX_POWER_DEMANDS
            || self
                .demands
                .values()
                .filter(|entry| entry.active && entry.demand.source == demand.source)
                .count()
                >= demand.source.capacity()
        {
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
        self.prune_reservation_tombstones(now_ms);
        if reservation.expires_at_ms <= now_ms {
            return Err(PowerReasonCode::Expired);
        }
        match self.reservations.get(&reservation.reservation_id) {
            Some(current) if current == &reservation => Ok(LedgerOutcome::Idempotent),
            Some(_) => Err(PowerReasonCode::DuplicateMismatch),
            None if self
                .reservation_tombstones
                .get(&reservation.reservation_id)
                .is_some_and(|current| current.reservation == reservation) =>
            {
                Ok(LedgerOutcome::Idempotent)
            }
            None if self
                .reservation_tombstones
                .contains_key(&reservation.reservation_id) =>
            {
                Err(PowerReasonCode::DuplicateMismatch)
            }
            None if self.reservations.len() + self.reservation_tombstones.len()
                >= MAX_POWER_DEMANDS =>
            {
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
        now_ms: u64,
    ) -> Result<LedgerOutcome, PowerReasonCode> {
        self.prune_reservation_tombstones(now_ms);
        if let Some(reservation) = self.reservations.remove(reservation_id) {
            self.insert_reservation_tombstone(reservation);
            return Ok(LedgerOutcome::Applied);
        }
        self.reservation_tombstones
            .contains_key(reservation_id)
            .then_some(LedgerOutcome::Idempotent)
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

    pub fn active_reservations(&mut self, now_ms: u64) -> impl Iterator<Item = &PowerReservation> {
        let expired: Vec<_> = self
            .reservations
            .values()
            .filter(|reservation| reservation.expires_at_ms <= now_ms)
            .cloned()
            .collect();
        for reservation in expired {
            self.reservations.remove(&reservation.reservation_id);
            self.insert_reservation_tombstone(reservation);
        }
        self.prune_reservation_tombstones(now_ms);
        self.reservations
            .values()
            .filter(move |item| item.not_before_ms <= now_ms && now_ms < item.expires_at_ms)
    }

    fn insert_reservation_tombstone(&mut self, reservation: PowerReservation) {
        self.reservation_tombstones.insert(
            reservation.reservation_id.clone(),
            ReservationTombstone {
                expires_at_ms: reservation.tombstone_expires_at_ms(),
                reservation,
            },
        );
    }

    fn prune_reservation_tombstones(&mut self, now_ms: u64) {
        self.reservation_tombstones
            .retain(|_, tombstone| tombstone.expires_at_ms > now_ms);
    }

    fn prune_expired_demands(&mut self, now_ms: u64) {
        self.demands
            .retain(|_, entry| entry.demand.expires_at_ms > now_ms);
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
    fn capacity_is_bounded_per_source_without_blocking_other_sources() {
        let mut ledger = DemandLedger::default();
        for index in 0..PowerDemandSource::Kws.capacity() {
            let mut item = demand(PowerDemandAction::Acquire, 1, 100);
            item.source = PowerDemandSource::Kws;
            item.demand_id = format!("f4f3e2d1-c0b9-48a7-9615-{index:012}");
            ledger.apply(item, 2).unwrap();
        }
        let mut overflow = demand(PowerDemandAction::Acquire, 1, 100);
        overflow.source = PowerDemandSource::Kws;
        overflow.demand_id = "f4f3e2d1-c0b9-48a7-9615-999999999999".into();
        assert_eq!(
            ledger.apply(overflow, 2),
            Err(PowerReasonCode::CapacityExceeded)
        );

        let mut ui = demand(PowerDemandAction::Acquire, 1, 100);
        ui.demand_id = "f4f3e2d1-c0b9-48a7-9615-888888888888".into();
        assert_eq!(ledger.apply(ui, 2), Ok(LedgerOutcome::Applied));
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

    #[test]
    fn released_demand_frees_source_capacity_for_a_new_id() {
        let mut ledger = DemandLedger::default();
        for index in 0..PowerDemandSource::Kws.capacity() {
            let mut item = demand(PowerDemandAction::Acquire, 1, 100);
            item.source = PowerDemandSource::Kws;
            item.demand_id = format!("f4f3e2d1-c0b9-48a7-9615-{index:012}");
            ledger.apply(item, 2).unwrap();
        }
        ledger
            .release("rover", "f4f3e2d1-c0b9-48a7-9615-000000000000")
            .unwrap();

        let mut replacement = demand(PowerDemandAction::Acquire, 1, 100);
        replacement.source = PowerDemandSource::Kws;
        replacement.demand_id = "f4f3e2d1-c0b9-48a7-9615-999999999999".into();
        assert_eq!(ledger.apply(replacement, 2), Ok(LedgerOutcome::Applied));
    }

    #[test]
    fn released_or_expired_reservation_cannot_be_changed_or_revived() {
        let mut ledger = DemandLedger::default();
        let reservation = PowerReservation {
            protocol_version: POWER_PROTOCOL_VERSION,
            reservation_id: "f4f3e2d1-c0b9-48a7-9615-141312111001".into(),
            role: LifecycleRole::Rover,
            entity_id: "rover".into(),
            authority: PowerAuthority {
                epoch: 1,
                sequence: 1,
            },
            required_profile: PowerProfile::ScheduledCapture,
            issued_at_ms: 1,
            not_before_ms: 1,
            expires_at_ms: 10,
        };
        ledger.register_reservation(reservation.clone(), 2).unwrap();
        assert_eq!(
            ledger.release_reservation(&reservation.reservation_id, 2),
            Ok(LedgerOutcome::Applied)
        );
        assert_eq!(
            ledger.register_reservation(reservation.clone(), 2),
            Ok(LedgerOutcome::Idempotent)
        );
        let mut changed = reservation.clone();
        changed.expires_at_ms = 20;
        assert_eq!(
            ledger.register_reservation(changed, 2),
            Err(PowerReasonCode::DuplicateMismatch)
        );

        let mut expired = reservation;
        expired.reservation_id = "f4f3e2d1-c0b9-48a7-9615-141312111002".into();
        ledger.register_reservation(expired.clone(), 2).unwrap();
        assert_eq!(ledger.active_reservations(10).count(), 0);
        assert_eq!(
            ledger.register_reservation(expired.clone(), 10),
            Err(PowerReasonCode::Expired)
        );
        let mut revived = expired.clone();
        revived.expires_at_ms = 20;
        assert_eq!(
            ledger.register_reservation(revived, 10),
            Err(PowerReasonCode::DuplicateMismatch)
        );
        let retained_until = expired.tombstone_expires_at_ms();
        let next = PowerReservation {
            reservation_id: "f4f3e2d1-c0b9-48a7-9615-141312111003".into(),
            issued_at_ms: retained_until,
            not_before_ms: retained_until,
            expires_at_ms: retained_until + 10,
            ..expired
        };
        assert_eq!(
            ledger.register_reservation(next, retained_until),
            Ok(LedgerOutcome::Applied)
        );
    }
}
