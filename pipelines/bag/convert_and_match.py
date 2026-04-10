import math

known_wrecks = {
    "Cedarville (Main)": (45 + 47.235/60.0, -(84 + 40.248/60.0)),
    "Cedarville (Crack)": (45 + 47.280/60.0, -(84 + 40.290/60.0)),
    "Cedarville (Stern)": (45 + 47.322/60.0, -(84 + 40.324/60.0)),
    "Young": (45 + 48.777/60.0, -(84 + 41.923/60.0))
}

targets = [
    (45.858929, -84.694687, 81633.4),
    (45.886430, -84.666721, 76983.4),
    (45.864179, -84.657732, 38405.6),
    (45.812362, -84.691039, 37716.7),
    (45.811811, -84.689280, 34961.1),
    (45.850487, -84.639546, 31000.0),
    (45.805292, -84.681944, 27038.9),
    (45.911264, -84.667873, 26005.6),
    (45.876824, -84.670897, 25488.9),
    (45.818916, -84.717196, 23594.5),
]

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # radius of Earth in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi/2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda/2.0)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

for name, (k_lat, k_lon) in known_wrecks.items():
    print(f"\n--- {name} ---")
    print(f"Known Coord: {k_lat:.6f}, {k_lon:.6f}")
    
    # Find closest target
    closest_t = None
    min_dist = float('inf')
    for t_lat, t_lon, t_size in targets:
        dist = haversine(k_lat, k_lon, t_lat, t_lon)
        if dist < min_dist:
            min_dist = dist
            closest_t = (t_lat, t_lon, t_size)
    
    if closest_t:
        print(f"Closest top-10 anomaly ({closest_t[2]:.1f} sqft) is at: {closest_t[0]:.6f}, {closest_t[1]:.6f}")
        print(f"Distance/Offset: {min_dist:.2f} meters")
        
        lat_offset = closest_t[0] - k_lat
        lon_offset = closest_t[1] - k_lon
        print(f"Lat Offset: {lat_offset:.6f} degrees")
        print(f"Lon Offset: {lon_offset:.6f} degrees")
