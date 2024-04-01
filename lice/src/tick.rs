use gc_arena::Collect;

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

#[derive(Debug, Default, Clone, Collect)]
#[collect(require_static)]
pub struct TickTable {
    table: Vec<TickInfo>,
}

impl TickTable {
    pub(crate) fn new() -> Self {
        Default::default()
    }

    pub(crate) fn find_entry(&self, name: &str) -> Option<Tick> {
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

    pub(crate) fn tick(&mut self, tick: Tick) {
        let entry = &mut self.table[tick.0];
        entry.count += 1;
        log::info!("encountered tick: {} = {}", entry.name, entry.count);
    }

    pub(crate) fn info(&self, tick: Tick) -> &TickInfo {
        &self.table[tick.0]
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Collect)]
#[collect(require_static)]
pub struct Tick(usize);
