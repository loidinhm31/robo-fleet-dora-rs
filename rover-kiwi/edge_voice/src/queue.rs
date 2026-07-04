use std::collections::VecDeque;

use robo_rover_lib::{TtsCommand, TtsPriority, VoiceReasonCode};

#[derive(Debug, Clone)]
pub struct QueuedCommand {
    pub command: TtsCommand,
    sequence: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EnqueueStatus {
    Accepted,
    Rejected(VoiceReasonCode),
}

#[derive(Debug, Clone)]
pub struct EnqueueOutcome {
    pub status: EnqueueStatus,
    pub interrupted_command_ids: Vec<String>,
}

#[derive(Debug)]
pub struct VoiceQueue {
    capacity: usize,
    next_sequence: u64,
    entries: VecDeque<QueuedCommand>,
}

impl VoiceQueue {
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "voice queue capacity must be positive");
        Self {
            capacity,
            next_sequence: 0,
            entries: VecDeque::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn enqueue(&mut self, command: TtsCommand) -> EnqueueOutcome {
        if command.priority == TtsPriority::Emergency {
            let interrupted_command_ids = self.clear_ids();
            self.push(command);
            return EnqueueOutcome {
                status: EnqueueStatus::Accepted,
                interrupted_command_ids,
            };
        }

        if self.entries.len() < self.capacity {
            self.push(command);
            return accepted();
        }

        if command.priority <= TtsPriority::Normal {
            return rejected(VoiceReasonCode::QueueFull);
        }

        if let Some(index) = self.lowest_priority_index_below(command.priority) {
            let interrupted_command_ids = self
                .entries
                .remove(index)
                .map(|entry| vec![entry.command.command_id])
                .unwrap_or_default();
            self.push(command);
            EnqueueOutcome {
                status: EnqueueStatus::Accepted,
                interrupted_command_ids,
            }
        } else {
            rejected(VoiceReasonCode::QueueFull)
        }
    }

    pub fn pop_next(&mut self) -> Option<TtsCommand> {
        let index = self
            .entries
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| {
                left.command
                    .priority
                    .cmp(&right.command.priority)
                    .then_with(|| right.sequence.cmp(&left.sequence))
            })
            .map(|(index, _)| index)?;
        self.entries.remove(index).map(|entry| entry.command)
    }

    pub fn clear_ids(&mut self) -> Vec<String> {
        self.entries
            .drain(..)
            .map(|entry| entry.command.command_id)
            .collect()
    }

    fn push(&mut self, command: TtsCommand) {
        let sequence = self.next_sequence;
        self.next_sequence = self.next_sequence.saturating_add(1);
        self.entries.push_back(QueuedCommand { command, sequence });
    }

    fn lowest_priority_index_below(&self, priority: TtsPriority) -> Option<usize> {
        self.entries
            .iter()
            .enumerate()
            .filter(|(_, entry)| entry.command.priority < priority)
            .min_by(|(_, left), (_, right)| {
                left.command
                    .priority
                    .cmp(&right.command.priority)
                    .then_with(|| left.sequence.cmp(&right.sequence))
            })
            .map(|(index, _)| index)
    }
}

fn accepted() -> EnqueueOutcome {
    EnqueueOutcome {
        status: EnqueueStatus::Accepted,
        interrupted_command_ids: Vec::new(),
    }
}

fn rejected(reason: VoiceReasonCode) -> EnqueueOutcome {
    EnqueueOutcome {
        status: EnqueueStatus::Rejected(reason),
        interrupted_command_ids: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn command(id: &str, priority: TtsPriority) -> TtsCommand {
        TtsCommand {
            command_id: id.to_string(),
            text: "hello".to_string(),
            timestamp: 1,
            priority,
        }
    }

    #[test]
    fn rejects_new_low_and_normal_when_full() {
        let mut queue = VoiceQueue::new(2);
        assert_eq!(
            queue.enqueue(command("a", TtsPriority::Low)).status,
            EnqueueStatus::Accepted
        );
        assert_eq!(
            queue.enqueue(command("b", TtsPriority::Normal)).status,
            EnqueueStatus::Accepted
        );
        assert_eq!(
            queue.enqueue(command("c", TtsPriority::Normal)).status,
            EnqueueStatus::Rejected(VoiceReasonCode::QueueFull)
        );
        assert_eq!(queue.len(), 2);
    }

    #[test]
    fn high_evicts_oldest_lower_priority_when_full() {
        let mut queue = VoiceQueue::new(2);
        queue.enqueue(command("low", TtsPriority::Low));
        queue.enqueue(command("normal", TtsPriority::Normal));
        let outcome = queue.enqueue(command("high", TtsPriority::High));
        assert_eq!(outcome.status, EnqueueStatus::Accepted);
        assert_eq!(outcome.interrupted_command_ids, vec!["low"]);
        assert_eq!(queue.pop_next().unwrap().command_id, "high");
        assert_eq!(queue.pop_next().unwrap().command_id, "normal");
    }

    #[test]
    fn emergency_clears_pending_commands() {
        let mut queue = VoiceQueue::new(3);
        queue.enqueue(command("a", TtsPriority::High));
        queue.enqueue(command("b", TtsPriority::Normal));
        let outcome = queue.enqueue(command("e", TtsPriority::Emergency));
        assert_eq!(outcome.interrupted_command_ids, vec!["a", "b"]);
        assert_eq!(queue.pop_next().unwrap().command_id, "e");
        assert!(queue.pop_next().is_none());
    }
}
