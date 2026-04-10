use anyhow::{Context, Result};
use chrono::{Datelike, NaiveDate, NaiveDateTime};
use serde::{Deserialize, Serialize};

use crate::config::{DetectionMethod, LakeConfig, WeatherWindow};

/// NOAA NDBC real-time data base URL.
/// Format: https://www.ndbc.noaa.gov/data/realtime2/{STATION_ID}.txt
const NDBC_REALTIME: &str = "https://www.ndbc.noaa.gov/data/realtime2";

/// Historical data (for building normalcy libraries over years).
/// Format: https://www.ndbc.noaa.gov/view_text_file.php?filename={STATION_ID}h{YEAR}.txt.gz&dir=data/historical/stdmet/
const NDBC_HISTORICAL: &str = "https://www.ndbc.noaa.gov/view_text_file.php";

// ── Parsed buoy observation ──────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuoyObs {
    pub station_id: String,
    pub timestamp: NaiveDateTime,
    pub wind_speed_mps: Option<f64>,    // m/s
    pub wind_dir_deg: Option<f64>,
    pub gust_mps: Option<f64>,
    pub wave_height_m: Option<f64>,
    pub dom_wave_period_s: Option<f64>,
    pub air_temp_c: Option<f64>,
    pub water_temp_c: Option<f64>,
    pub pressure_hpa: Option<f64>,
    pub visibility_nm: Option<f64>,
}

impl BuoyObs {
    /// Wind speed in mph (for easier threshold comparison).
    pub fn wind_mph(&self) -> Option<f64> {
        self.wind_speed_mps.map(|v| v * 2.237)
    }

    /// Is this a calm observation? (wind < 5 mph)
    pub fn is_calm(&self) -> bool {
        self.wind_mph().map_or(false, |w| w < 5.0)
    }

    /// Is wind from N/NW sector? (315° ± 67.5° = 247.5° to 22.5°)
    pub fn is_nnw_wind(&self) -> bool {
        self.wind_dir_deg.map_or(false, |d| d >= 292.5 || d <= 22.5)
    }
}

// ── Weather condition assessment ─────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeatherAssessment {
    pub station_id: String,
    pub assessed_at: NaiveDateTime,
    pub window: Option<WeatherWindow>,
    pub confidence: f64,       // 0.0–1.0
    pub reason: String,
    pub observations_used: usize,
}

// ── Weather client ───────────────────────────────────────────────────

pub struct WeatherClient {
    http: reqwest::Client,
}

impl WeatherClient {
    pub fn new() -> Self {
        Self {
            http: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .build()
                .expect("http client"),
        }
    }

    /// Fetch recent observations (last 45 days) for a NOAA NDBC buoy.
    pub async fn fetch_recent(&self, station_id: &str) -> Result<Vec<BuoyObs>> {
        let url = format!("{NDBC_REALTIME}/{station_id}.txt");
        tracing::info!("fetching buoy data: {url}");

        let resp = self.http.get(&url).send().await
            .context(format!("fetch buoy {station_id}"))?;

        if !resp.status().is_success() {
            anyhow::bail!("buoy {station_id} returned {}", resp.status());
        }

        let text = resp.text().await?;
        parse_ndbc_stdmet(&text, station_id)
    }

    /// Fetch historical observations for a given year.
    pub async fn fetch_historical(
        &self,
        station_id: &str,
        year: i32,
    ) -> Result<Vec<BuoyObs>> {
        let url = format!(
            "{NDBC_HISTORICAL}?filename={station_id}h{year}.txt.gz&dir=data/historical/stdmet/"
        );
        tracing::info!("fetching historical buoy data: {url}");

        let resp = self.http.get(&url).send().await
            .context(format!("fetch buoy {station_id} year {year}"))?;

        if !resp.status().is_success() {
            anyhow::bail!("buoy {station_id} year {year} returned {}", resp.status());
        }

        let text = resp.text().await?;
        parse_ndbc_stdmet(&text, station_id)
    }

    /// Assess current weather conditions at a lake for a specific detection method.
    pub async fn assess_conditions(
        &self,
        lake: &LakeConfig,
        method: DetectionMethod,
    ) -> Result<WeatherAssessment> {
        let target_window = WeatherWindow::for_method(method);

        // Fetch from primary buoy (first in list)
        let buoy = lake.buoys.first()
            .ok_or_else(|| anyhow::anyhow!("no buoys for {}", lake.lake))?;

        let obs = self.fetch_recent(&buoy.id).await?;
        if obs.is_empty() {
            return Ok(WeatherAssessment {
                station_id: buoy.id.clone(),
                assessed_at: chrono::Utc::now().naive_utc(),
                window: None,
                confidence: 0.0,
                reason: "no recent observations".into(),
                observations_used: 0,
            });
        }

        match target_window {
            WeatherWindow::TransitionDay => assess_transition(&obs, &buoy.id),
            WeatherWindow::CalmSummer => assess_calm(&obs, &buoy.id),
            WeatherWindow::PostStorm => assess_post_storm(&obs, &buoy.id),
        }
    }

    /// Find historical dates matching a weather window for a given lake.
    /// Useful for selecting training data from satellite archives.
    pub async fn find_historical_windows(
        &self,
        lake: &LakeConfig,
        method: DetectionMethod,
        year: i32,
    ) -> Result<Vec<NaiveDate>> {
        let buoy = lake.buoys.first()
            .ok_or_else(|| anyhow::anyhow!("no buoys for {}", lake.lake))?;

        let obs = self.fetch_historical(&buoy.id, year).await?;
        let target_window = WeatherWindow::for_method(method);

        let matching_dates = find_window_dates(&obs, target_window);
        Ok(matching_dates)
    }
}

// ── NDBC text parsing ────────────────────────────────────────────────
// NDBC stdmet files: header lines start with #, then 2 more header rows,
// then data rows. Columns are fixed-width, space-separated.
// #YY  MM DD hh mm WDIR WSPD GST  WVHT   DPD   APD MWD   PRES  ATMP  WTMP  DEWP  VIS PTDY  TIDE
// #yr  mo dy hr mn deg  m/s  m/s     m   sec   sec deg    hPa  degC  degC  degC   nmi  hPa    ft

fn parse_ndbc_stdmet(text: &str, station_id: &str) -> Result<Vec<BuoyObs>> {
    let mut obs = Vec::new();

    for line in text.lines() {
        if line.starts_with('#') || line.is_empty() {
            continue;
        }

        let cols: Vec<&str> = line.split_whitespace().collect();
        if cols.len() < 15 {
            continue;
        }

        let year: i32 = cols[0].parse().unwrap_or(0);
        let month: u32 = cols[1].parse().unwrap_or(0);
        let day: u32 = cols[2].parse().unwrap_or(0);
        let hour: u32 = cols[3].parse().unwrap_or(0);
        let minute: u32 = cols[4].parse().unwrap_or(0);

        let dt = match NaiveDate::from_ymd_opt(year, month, day)
            .and_then(|d| d.and_hms_opt(hour, minute, 0))
        {
            Some(dt) => dt,
            None => continue,
        };

        let parse_f64 = |s: &str| -> Option<f64> {
            let v: f64 = s.parse().ok()?;
            // NDBC uses 99, 999, 9999 etc. as missing values
            if v >= 99.0 && (v - v.round()).abs() < 0.001 {
                // Check if it's a missing-value sentinel
                let rounded = v.round() as i64;
                if rounded == 99 || rounded == 999 || rounded == 9999 {
                    return None;
                }
            }
            Some(v)
        };

        obs.push(BuoyObs {
            station_id: station_id.to_string(),
            timestamp: dt,
            wind_dir_deg: parse_f64(cols[5]),
            wind_speed_mps: parse_f64(cols[6]),
            gust_mps: parse_f64(cols[7]),
            wave_height_m: parse_f64(cols[8]),
            dom_wave_period_s: parse_f64(cols[9]),
            // cols[10] = APD, cols[11] = MWD
            air_temp_c: parse_f64(cols[13]),
            water_temp_c: parse_f64(cols[14]),
            pressure_hpa: parse_f64(cols[12]),
            visibility_nm: if cols.len() > 16 { parse_f64(cols[16]) } else { None },
        });
    }

    Ok(obs)
}

// ── Window classifiers ───────────────────────────────────────────────

/// TransitionDay: wind was ≥ 15 mph within last 12h but is now ≤ 5 mph.
fn assess_transition(obs: &[BuoyObs], station: &str) -> Result<WeatherAssessment> {
    if obs.len() < 2 {
        return Ok(WeatherAssessment {
            station_id: station.into(),
            assessed_at: chrono::Utc::now().naive_utc(),
            window: None,
            confidence: 0.0,
            reason: "insufficient observations".into(),
            observations_used: obs.len(),
        });
    }

    // Look at last 12 hours of data (observations are typically hourly)
    let latest = &obs[0]; // most recent
    let lookback_hours = 12;
    let cutoff = latest.timestamp - chrono::Duration::hours(lookback_hours);

    let recent: Vec<&BuoyObs> = obs.iter()
        .filter(|o| o.timestamp >= cutoff)
        .collect();

    let current_wind = latest.wind_mph().unwrap_or(99.0);
    let max_wind = recent.iter()
        .filter_map(|o| o.wind_mph())
        .fold(0.0_f64, f64::max);

    let is_transition = current_wind <= 5.0 && max_wind >= 15.0;
    let confidence = if is_transition {
        // Higher confidence if wind drop was recent and dramatic
        let drop = max_wind - current_wind;
        (drop / 20.0).min(1.0)
    } else {
        0.0
    };

    Ok(WeatherAssessment {
        station_id: station.into(),
        assessed_at: chrono::Utc::now().naive_utc(),
        window: if is_transition { Some(WeatherWindow::TransitionDay) } else { None },
        confidence,
        reason: format!(
            "current {:.0} mph, 12h max {:.0} mph{}",
            current_wind, max_wind,
            if is_transition { " → TRANSITION" } else { "" }
        ),
        observations_used: recent.len(),
    })
}

/// CalmSummer: sustained winds < 5 mph, no recent precip, July–October.
fn assess_calm(obs: &[BuoyObs], station: &str) -> Result<WeatherAssessment> {
    let latest = match obs.first() {
        Some(o) => o,
        None => return Ok(WeatherAssessment {
            station_id: station.into(),
            assessed_at: chrono::Utc::now().naive_utc(),
            window: None,
            confidence: 0.0,
            reason: "no observations".into(),
            observations_used: 0,
        }),
    };

    // Check season (July–October = months 7–10)
    let month = latest.timestamp.date().month();
    if !(7..=10).contains(&month) {
        return Ok(WeatherAssessment {
            station_id: station.into(),
            assessed_at: chrono::Utc::now().naive_utc(),
            window: None,
            confidence: 0.0,
            reason: format!("month {} outside Jul-Oct window", month),
            observations_used: 1,
        });
    }

    // Check last 6h sustained calm
    let cutoff = latest.timestamp - chrono::Duration::hours(6);
    let recent: Vec<&BuoyObs> = obs.iter()
        .filter(|o| o.timestamp >= cutoff)
        .collect();

    let all_calm = recent.iter().all(|o| o.is_calm());
    let avg_wind: f64 = recent.iter()
        .filter_map(|o| o.wind_mph())
        .sum::<f64>() / recent.len().max(1) as f64;

    let confidence = if all_calm { (1.0 - avg_wind / 5.0).max(0.0) } else { 0.0 };

    Ok(WeatherAssessment {
        station_id: station.into(),
        assessed_at: chrono::Utc::now().naive_utc(),
        window: if all_calm { Some(WeatherWindow::CalmSummer) } else { None },
        confidence,
        reason: format!(
            "6h avg wind {:.1} mph, all calm={}",
            avg_wind, all_calm
        ),
        observations_used: recent.len(),
    })
}

/// PostStorm: had ≥24h of N/NW winds ≥20 mph, now clearing (wind dropping).
fn assess_post_storm(obs: &[BuoyObs], station: &str) -> Result<WeatherAssessment> {
    let latest = match obs.first() {
        Some(o) => o,
        None => return Ok(WeatherAssessment {
            station_id: station.into(),
            assessed_at: chrono::Utc::now().naive_utc(),
            window: None,
            confidence: 0.0,
            reason: "no observations".into(),
            observations_used: 0,
        }),
    };

    // Look back 48 hours to find a storm window
    let cutoff_48h = latest.timestamp - chrono::Duration::hours(48);
    let recent_48h: Vec<&BuoyObs> = obs.iter()
        .filter(|o| o.timestamp >= cutoff_48h)
        .collect();

    // Find consecutive N/NW wind ≥ 20 mph periods
    let mut storm_hours = 0u32;
    let mut max_storm_hours = 0u32;
    let mut storm_end_time: Option<NaiveDateTime> = None;

    for pair in recent_48h.windows(2) {
        let o = pair[0];
        let is_nnw_storm = o.is_nnw_wind() && o.wind_mph().unwrap_or(0.0) >= 20.0;

        if is_nnw_storm {
            storm_hours += 1;
            if storm_hours > max_storm_hours {
                max_storm_hours = storm_hours;
                // The storm ended at the next observation
                storm_end_time = Some(pair[1].timestamp);
            }
        } else {
            storm_hours = 0;
        }
    }

    let current_wind = latest.wind_mph().unwrap_or(99.0);
    let had_storm = max_storm_hours >= 24;
    let is_clearing = current_wind < 15.0;
    let is_post_storm = had_storm && is_clearing;

    let hours_since_storm = storm_end_time
        .map(|t| (latest.timestamp - t).num_hours())
        .unwrap_or(999);

    // Best confidence: storm ended 6-24h ago, wind now < 10 mph
    let confidence = if is_post_storm {
        let time_factor = if (6..=24).contains(&hours_since_storm) { 1.0 }
            else if hours_since_storm < 6 { 0.5 }  // still settling
            else { (1.0 - (hours_since_storm - 24) as f64 / 48.0).max(0.1) };
        let wind_factor = (1.0 - current_wind / 15.0).max(0.0);
        time_factor * wind_factor
    } else {
        0.0
    };

    Ok(WeatherAssessment {
        station_id: station.into(),
        assessed_at: chrono::Utc::now().naive_utc(),
        window: if is_post_storm { Some(WeatherWindow::PostStorm) } else { None },
        confidence,
        reason: format!(
            "max N/NW storm {}h, current wind {:.0} mph, hours since storm: {}",
            max_storm_hours, current_wind, hours_since_storm
        ),
        observations_used: recent_48h.len(),
    })
}

/// Scan historical observations to find dates matching a weather window.
fn find_window_dates(obs: &[BuoyObs], window: WeatherWindow) -> Vec<NaiveDate> {
    let mut dates = Vec::new();
    let mut seen = std::collections::HashSet::new();

    // Group observations by date
    for o in obs {
        let date = o.timestamp.date();
        if seen.contains(&date) {
            continue;
        }

        let matches = match window {
            WeatherWindow::TransitionDay => {
                // Simplified: wind dropped ≥10 mph within this day
                let day_obs: Vec<&BuoyObs> = obs.iter()
                    .filter(|ob| ob.timestamp.date() == date)
                    .collect();
                let max_w = day_obs.iter().filter_map(|ob| ob.wind_mph()).fold(0.0_f64, f64::max);
                let min_w = day_obs.iter().filter_map(|ob| ob.wind_mph()).fold(99.0_f64, f64::min);
                max_w >= 15.0 && min_w <= 5.0
            }
            WeatherWindow::CalmSummer => {
                let month = date.month();
                if !(7..=10).contains(&month) { false }
                else {
                    let day_obs: Vec<&BuoyObs> = obs.iter()
                        .filter(|ob| ob.timestamp.date() == date)
                        .collect();
                    day_obs.iter().all(|ob| ob.is_calm()) && day_obs.len() >= 4
                }
            }
            WeatherWindow::PostStorm => {
                // Check if yesterday had N/NW ≥ 20 mph but today is calmer
                let yesterday = date - chrono::Duration::days(1);
                let yest_obs: Vec<&BuoyObs> = obs.iter()
                    .filter(|ob| ob.timestamp.date() == yesterday)
                    .collect();
                let today_obs: Vec<&BuoyObs> = obs.iter()
                    .filter(|ob| ob.timestamp.date() == date)
                    .collect();

                let had_strong_nnw = yest_obs.iter().any(|ob|
                    ob.is_nnw_wind() && ob.wind_mph().unwrap_or(0.0) >= 20.0
                );
                let today_calmer = today_obs.iter().any(|ob|
                    ob.wind_mph().unwrap_or(99.0) < 15.0
                );
                had_strong_nnw && today_calmer
            }
        };

        if matches {
            dates.push(date);
            seen.insert(date);
        }
    }

    dates
}
