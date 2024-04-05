#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, gc_arena::Collect)]
#[collect(require_static)]
pub struct Tick(usize);

#[derive(Debug, Default, Clone)]
pub struct TickInfo {
    pub name: String,
    pub count: usize,
}

impl TickInfo {
    fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            count: 0,
        }
    }
}

#[derive(Debug, Default, Clone, gc_arena::Collect)]
#[collect(require_static)]
pub struct TickTable {
    table: Vec<TickInfo>,
}

impl TickTable {
    pub(crate) fn new() -> Self {
        Default::default()
    }

    fn find_entry(&self, name: &str) -> Option<Tick> {
        Some(Tick(
            self.table
                .iter()
                .enumerate()
                .find(|(_, s)| s.name == name)?
                .0,
        ))
    }

    pub(crate) fn add_entry(&mut self, name: &str) -> Tick {
        self.find_entry(name).unwrap_or_else(|| {
            self.table.push(TickInfo::new(name));
            let index = self.table.len() - 1;
            log::debug!("added tick: {} (index = {})", name, index);
            Tick(index)
        })
    }

    pub(crate) fn tick(&mut self, tick: Tick) -> Result<&TickInfo, TickError> {
        let entry = self
            .table
            .get_mut(tick.0)
            .ok_or(TickError::MissingEntry(tick.0))?;
        entry.count += 1;
        log::info!("encountered tick: {} = {}", entry.name, entry.count);
        Ok(entry)
    }

    pub(crate) fn info(&self, tick: Tick) -> &TickInfo {
        &self.table[tick.0]
    }
}

#[derive(Debug, thiserror::Error)]
pub enum TickError {
    #[error("could not find tick entry at index {0}")]
    MissingEntry(usize),
}
