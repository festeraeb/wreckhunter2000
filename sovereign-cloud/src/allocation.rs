use anyhow::Result;
use nauticuvs::protocol::{NodeCapabilities, NodeRole};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn};

pub struct AllocationEngine {
    pub capabilities: Arc<RwLock<NodeCapabilities>>,
}

impl AllocationEngine {
    pub async fn detect() -> Result<Self> {
        let node_id = hostname::get()
            .map(|h| h.to_string_lossy().to_string())
            .unwrap_or_else(|_| uuid::Uuid::new_v4().to_string());

        let (total_vram_gb, available_vram_gb, gpu_name, has_fp64) = Self::query_hardware();
        let has_tpu = Self::detect_tpu();

        let caps = NodeCapabilities {
            node_id,
            total_vram_gb,
            available_vram_gb,
            has_fp64,
            has_tpu,
            gpu_name,
        };

        info!(
            "Hardware: {} | GPU: {} | VRAM: {}GB total / {}GB free | FP64: {} | TPU: {}",
            caps.node_id, caps.gpu_name, caps.total_vram_gb, caps.available_vram_gb,
            caps.has_fp64, caps.has_tpu
        );

        Ok(Self {
            capabilities: Arc::new(RwLock::new(caps)),
        })
    }

    fn query_hardware() -> (u32, u32, String, bool) {
        if let Ok(result) = Self::query_nvml() {
            return result;
        }
        Self::query_wgpu_adapters().unwrap_or_else(|| {
            warn!("No GPU detected — running CPU-only mode");
            (0, 0, "CPU-only".to_string(), false)
        })
    }

    fn query_nvml() -> Result<(u32, u32, String, bool)> {
        use nvml_wrapper::Nvml;
        let nvml = Nvml::init()?;
        let device = nvml.device_by_index(0)?;

        let mem = device.memory_info()?;
        let total_gb = (mem.total / (1024 * 1024 * 1024)) as u32;
        let avail_gb = (mem.free / (1024 * 1024 * 1024)) as u32;
        let name = device.name()?;

        // Professional/compute cards with native FP64 support
        let has_fp64 = ["P100", "P40", "P4", "V100", "A100", "A40", "Quadro", "Tesla"]
            .iter()
            .any(|tag| name.contains(tag));

        info!("NVML: {} | {}GB total / {}GB free | FP64: {}", name, total_gb, avail_gb, has_fp64);
        Ok((total_gb, avail_gb, name, has_fp64))
    }

    fn query_wgpu_adapters() -> Option<(u32, u32, String, bool)> {
        use wgpu::{Backends, Instance, PowerPreference, RequestAdapterOptions};

        // wgpu 29.x: use default InstanceDescriptor and set backends via backend_options
        let _ = Backends::all(); // keep the import used
        let instance = Instance::default();

        let adapter = pollster::block_on(instance.request_adapter(&RequestAdapterOptions {
            power_preference: PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })).ok()?;

        let info = adapter.get_info();
        let gpu_name = info.name.clone();
        let vram_gb = Self::estimate_vram_from_name(&gpu_name);
        // Conservative: only discrete GPUs may have FP64
        let has_fp64 = info.device_type == wgpu::DeviceType::DiscreteGpu
            && ["P100", "P40", "Quadro", "Tesla"].iter().any(|t| gpu_name.contains(t));

        warn!("NVML unavailable — wgpu fallback: {} (~{}GB estimated VRAM)", gpu_name, vram_gb);
        Some((vram_gb, vram_gb, gpu_name, has_fp64))
    }

    fn estimate_vram_from_name(name: &str) -> u32 {
        let n = name.to_uppercase();
        if n.contains("P40") { 24 }
        else if n.contains("P100") { 16 }
        else if n.contains("P4") { 8 }
        else if n.contains("1080 TI") || n.contains("1080TI") { 11 }
        else if n.contains("1080") { 8 }
        else if n.contains("1070") { 8 }
        else if n.contains("1060") { 6 }
        else if n.contains("3090") { 24 }
        else if n.contains("3080 TI") || n.contains("3080TI") { 12 }
        else if n.contains("3080") { 10 }
        else if n.contains("4090") { 24 }
        else if n.contains("4080") { 16 }
        else if n.contains("4070 TI") || n.contains("4070TI") { 12 }
        else { 4 }
    }

    fn detect_tpu() -> bool {
        // Coral TPU on Linux appears as /dev/apex_0 (M.2/PCIe) or /dev/accel0
        std::path::Path::new("/dev/apex_0").exists()
            || std::path::Path::new("/dev/accel0").exists()
    }

    pub async fn determine_role(&self) -> NodeRole {
        let caps = self.capabilities.read().await;
        match caps.total_vram_gb {
            v if v >= 24 => NodeRole::SuperAgent,
            v if v >= 8 && caps.has_fp64 => NodeRole::Analyst,
            v if v >= 12 => NodeRole::CoderReasoningLead,
            v if v >= 8 => NodeRole::CoderReasoningLead,
            v if v >= 6 => NodeRole::CoderExecutionWorker,
            _ if caps.has_tpu => NodeRole::Scout,
            _ => NodeRole::Idle,
        }
    }

    pub async fn can_accept(&self, required_vram_gb: u32, required_fp64: bool, requires_tpu: bool) -> bool {
        let caps = self.capabilities.read().await;
        caps.available_vram_gb >= required_vram_gb
            && (!required_fp64 || caps.has_fp64)
            && (!requires_tpu || caps.has_tpu)
    }

    pub async fn reserve(&self, vram_gb: u32) {
        let mut caps = self.capabilities.write().await;
        caps.available_vram_gb = caps.available_vram_gb.saturating_sub(vram_gb);
    }

    pub async fn release(&self, vram_gb: u32) {
        let mut caps = self.capabilities.write().await;
        caps.available_vram_gb = (caps.available_vram_gb + vram_gb).min(caps.total_vram_gb);
    }
}
