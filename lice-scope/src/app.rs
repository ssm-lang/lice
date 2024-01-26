use std::{fs::File, io::Read, path::PathBuf, process, str::FromStr};

use eframe::{run_native, App, CreationContext};
use egui::Context;
use egui_graphs::{GraphView, SettingsInteraction, SettingsNavigation, SettingsStyle};
use lice::comb::CombFile;
use log::error;

use crate::gui::{to_gui_graph, EdgeShape, GuiGraph, NodeShape};

pub struct CombApp {
    g: GuiGraph,
}

impl CombApp {
    pub fn new(_: &CreationContext<'_>, c: CombFile) -> Self {
        Self {
            g: to_gui_graph(&c.program),
        }
    }

    pub fn run(filename: PathBuf) {
        let Ok(mut f) = File::open(&filename) else {
            error!("No such file or directory: {}", filename.to_string_lossy());
            process::exit(1);
        };
        let mut buf = String::new();
        f.read_to_string(&mut buf).unwrap_or_else(|e| {
            error!("{e}");
            process::exit(1);
        });
        let c = CombFile::from_str(&buf).unwrap_or_else(|e| {
            error!("{e}");
            process::exit(1);
        });

        let native_options = eframe::NativeOptions::default();
        run_native(
            filename.to_str().unwrap(),
            native_options,
            Box::new(|cc| Box::new(CombApp::new(cc, c))),
        )
        .unwrap();
    }
}

impl App for CombApp {
    fn update(&mut self, ctx: &Context, _: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.add(
                &mut GraphView::<_, _, _, _, NodeShape, EdgeShape>::new(&mut self.g)
                    .with_interactions(&SettingsInteraction::new().with_dragging_enabled(true))
                    .with_styles(&SettingsStyle::new().with_labels_always(true))
                    .with_navigations(
                        &SettingsNavigation::new()
                            .with_zoom_and_pan_enabled(true)
                            .with_fit_to_screen_enabled(false),
                    ),
            );
        });
    }
}
