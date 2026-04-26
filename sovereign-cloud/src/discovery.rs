use anyhow::Result;
use mdns_sd::{ServiceDaemon, ServiceEvent, ServiceInfo};
use nauticuvs::protocol::NodeCapabilities;
use std::collections::HashMap;
use std::net::{IpAddr, Ipv4Addr, UdpSocket};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{error, info};

const SERVICE_TYPE: &str = "_cesarops._tcp.local.";

pub struct NodeDiscovery {
    daemon: ServiceDaemon,
    peers: Arc<RwLock<HashMap<String, NodeCapabilities>>>,
}

impl NodeDiscovery {
    pub fn new() -> Result<Self> {
        let daemon = ServiceDaemon::new()?;
        Ok(Self {
            daemon,
            peers: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    pub fn announce(&self, caps: &NodeCapabilities, port: u16) -> Result<()> {
        let host_ip: IpAddr = local_ipv4()
            .map(IpAddr::V4)
            .unwrap_or(IpAddr::V4(Ipv4Addr::LOCALHOST));
        // mDNS hostnames must not contain dots other than the .local. suffix
        let safe_name = caps.node_id.replace('.', "-");
        let hostname = format!("{}.local.", safe_name);

        let props: HashMap<String, String> = [
            ("vram_gb".to_string(), caps.total_vram_gb.to_string()),
            ("fp64".to_string(), caps.has_fp64.to_string()),
            ("tpu".to_string(), caps.has_tpu.to_string()),
            ("gpu".to_string(), caps.gpu_name.clone()),
            ("port".to_string(), port.to_string()),
        ]
        .into_iter()
        .collect();

        let service = ServiceInfo::new(
            SERVICE_TYPE,
            &safe_name,
            &hostname,
            host_ip,
            port,
            Some(props),
        )?;

        self.daemon.register(service)?;
        info!("mDNS: announced {} ({}) on port {}", safe_name, host_ip, port);
        Ok(())
    }

    pub async fn browse_peers(&self) -> Result<()> {
        let receiver = self.daemon.browse(SERVICE_TYPE)?;
        let peers = self.peers.clone();

        tokio::spawn(async move {
            loop {
                match receiver.recv_async().await {
                    Ok(ServiceEvent::ServiceResolved(info)) => {
                        let props = info.get_properties();
                        let vram_gb = props
                            .get("vram_gb")
                            .and_then(|p| p.val_str().parse::<u32>().ok())
                            .unwrap_or(0);
                        let has_fp64 = props
                            .get("fp64")
                            .and_then(|p| p.val_str().parse::<bool>().ok())
                            .unwrap_or(false);
                        let has_tpu = props
                            .get("tpu")
                            .and_then(|p| p.val_str().parse::<bool>().ok())
                            .unwrap_or(false);
                        let gpu_name = props
                            .get("gpu")
                            .map(|p| p.val_str().to_string())
                            .unwrap_or_default();

                        let peer = NodeCapabilities {
                            node_id: info.get_fullname().to_string(),
                            total_vram_gb: vram_gb,
                            available_vram_gb: vram_gb,
                            has_fp64,
                            has_tpu,
                            gpu_name,
                        };

                        info!("mDNS: discovered peer {} ({}GB VRAM, FP64={}, TPU={})",
                            peer.node_id, vram_gb, has_fp64, has_tpu);
                        peers.write().await.insert(peer.node_id.clone(), peer);
                    }
                    Ok(ServiceEvent::ServiceRemoved(_, fullname)) => {
                        info!("mDNS: peer departed — {}", fullname);
                        peers.write().await.remove(&fullname);
                    }
                    Ok(_) => {}
                    Err(e) => {
                        error!("mDNS browse error: {}", e);
                        break;
                    }
                }
            }
        });

        Ok(())
    }

    pub async fn get_peers(&self) -> Vec<NodeCapabilities> {
        self.peers.read().await.values().cloned().collect()
    }

    /// Find the best peer for a task requiring specific resources.
    pub async fn find_peer_for(
        &self,
        required_vram_gb: u32,
        required_fp64: bool,
        requires_tpu: bool,
    ) -> Option<NodeCapabilities> {
        let peers = self.peers.read().await;
        peers
            .values()
            .filter(|p| {
                p.available_vram_gb >= required_vram_gb
                    && (!required_fp64 || p.has_fp64)
                    && (!requires_tpu || p.has_tpu)
            })
            .max_by_key(|p| p.available_vram_gb)
            .cloned()
    }
}

fn local_ipv4() -> Option<Ipv4Addr> {
    let socket = UdpSocket::bind("0.0.0.0:0").ok()?;
    socket.connect("8.8.8.8:80").ok()?;
    match socket.local_addr().ok()? {
        std::net::SocketAddr::V4(addr) => Some(*addr.ip()),
        _ => None,
    }
}
