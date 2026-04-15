use anyhow::Result;
use clap::Parser;
use comfy_table::Table;
use gguf_rs::{get_gguf_container, GGMLType, GGUFModel, MetadataValue};
use log::LevelFilter;
use simple_logger::SimpleLogger;

#[derive(Parser)]
#[command(author, version, about)]
struct Cli {
    /// model file path
    model: String,

    /// show metadata
    #[arg(short, long, default_value_t = true)]
    metadata: bool,

    /// show tensors
    #[arg(short, long, default_value_t = false)]
    tensors: bool,

    /// show debug log
    #[arg(short, long)]
    debug: bool,
}

fn print_metadata(model: &GGUFModel) {
    let mut table = Table::new();
    table.set_header(vec!["#", "Key", "Value"]);

    fn value_to_string_recursive(value: &MetadataValue, fuel: usize) -> String {
        if fuel == 0 {
            return "...".to_owned();
        }

        match value {
            MetadataValue::Bool(v) => v.to_string(),
            MetadataValue::Uint8(v) => v.to_string(),
            MetadataValue::Int8(v) => v.to_string(),
            MetadataValue::Uint16(v) => v.to_string(),
            MetadataValue::Int16(v) => v.to_string(),
            MetadataValue::Uint32(v) => v.to_string(),
            MetadataValue::Int32(v) => v.to_string(),
            MetadataValue::Uint64(v) => v.to_string(),
            MetadataValue::Int64(v) => v.to_string(),
            MetadataValue::Float32(v) => v.to_string(),
            MetadataValue::Float64(v) => v.to_string(),
            MetadataValue::String(v) => v.to_owned(),
            MetadataValue::Array(v) => {
                let concat_values = v
                    .iter()
                    .map(|v| value_to_string_recursive(v, fuel - 1))
                    .collect::<Vec<String>>()
                    .join(",");
                format!("[{}]", concat_values)
            }
        }
    }

    const MAX_RECURSIVE_PRINTS: usize = 4;

    model
        .metadata()
        .iter()
        .enumerate()
        .for_each(|(i, (key, value))| {
            let unwrap_value = value_to_string_recursive(value, MAX_RECURSIVE_PRINTS);
            table.add_row(vec![(i + 1).to_string(), key.clone(), unwrap_value]);
        });

    println!("Metadata:\n{table}");
}

fn print_tensors(model: &GGUFModel) {
    let mut table = Table::new();
    table.set_header(vec!["#", "Name", "Type", "Dimension", "Offset"]);

    model.tensors().iter().enumerate().for_each(|(i, tensor)| {
        table.add_row(vec![
            (i + 1).to_string(),
            tensor.name.clone(),
            GGMLType::try_from(tensor.kind).unwrap().to_string(),
            tensor
                .shape
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(","),
            tensor.offset.to_string(),
        ]);
    });

    println!("Tensors:\n{table}");
}

fn main() -> Result<()> {
    let args = Cli::parse();

    if args.debug {
        SimpleLogger::default()
            .with_level(LevelFilter::Debug)
            .init()
            .unwrap();
    }

    let mut gguf_container = get_gguf_container(&args.model)?;
    let gguf_model = gguf_container.decode()?;

    if args.metadata {
        print_metadata(&gguf_model);
    }

    if args.tensors {
        print_tensors(&gguf_model);
    }
    Ok(())
}
