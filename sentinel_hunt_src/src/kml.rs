//! KML/KMZ generation for satellite detection results.
//!
//! Produces Google-Earth-compatible files with detection pins,
//! search area polygons, and orbit-track swaths.

use std::io::Write;
use std::path::Path;

use anyhow::Result;
use quick_xml::events::{BytesDecl, BytesEnd, BytesStart, BytesText, Event};
use quick_xml::Writer;

use crate::config::{DetectionMethod, Lake};
use crate::detect::{Detection, DetectionResult};

// ── Style constants ──────────────────────────────────────────────────

fn method_color(method: DetectionMethod) -> &'static str {
    // KML colors: aaBBGGRR (alpha, blue, green, red)
    match method {
        DetectionMethod::DarkSpot => "ff0000ff",       // red
        DetectionMethod::ClearHole => "ffff8800",      // blue-ish (cyan)
        DetectionMethod::SedimentTrap => "ff00aaff",   // orange
    }
}

fn method_icon(method: DetectionMethod) -> &'static str {
    match method {
        DetectionMethod::DarkSpot => "http://maps.google.com/mapfiles/kml/shapes/target.png",
        DetectionMethod::ClearHole => "http://maps.google.com/mapfiles/kml/shapes/water.png",
        DetectionMethod::SedimentTrap => "http://maps.google.com/mapfiles/kml/shapes/volcano.png",
    }
}

// ── Public API ───────────────────────────────────────────────────────

/// Write detection results to a KML file.
pub fn write_kml(results: &[DetectionResult], path: &Path) -> Result<()> {
    let mut buf = Vec::new();
    generate_kml(results, &mut buf)?;
    std::fs::write(path, &buf)?;
    Ok(())
}

/// Write detection results to a KMZ (zipped KML) file.
pub fn write_kmz(results: &[DetectionResult], path: &Path) -> Result<()> {
    let mut kml_buf = Vec::new();
    generate_kml(results, &mut kml_buf)?;

    let file = std::fs::File::create(path)?;
    let mut zip = zip::ZipWriter::new(file);
    let options = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated);
    zip.start_file("doc.kml", options)?;
    zip.write_all(&kml_buf)?;
    zip.finish()?;
    Ok(())
}

/// Generate a summary KML string (for embedding or preview).
pub fn generate_kml_string(results: &[DetectionResult]) -> Result<String> {
    let mut buf = Vec::new();
    generate_kml(results, &mut buf)?;
    Ok(String::from_utf8(buf)?)
}

// ── KML generation ───────────────────────────────────────────────────

fn generate_kml<W: Write>(results: &[DetectionResult], writer: W) -> Result<()> {
    let mut xml = Writer::new_with_indent(writer, b' ', 2);

    // XML declaration
    xml.write_event(Event::Decl(BytesDecl::new("1.0", Some("UTF-8"), None)))?;

    // <kml>
    let mut kml = BytesStart::new("kml");
    kml.push_attribute(("xmlns", "http://www.opengis.net/kml/2.2"));
    xml.write_event(Event::Start(kml))?;

    // <Document>
    xml.write_event(Event::Start(BytesStart::new("Document")))?;
    write_text_element(&mut xml, "name", "Sentinel Hunt — Wreck Detections")?;
    write_text_element(
        &mut xml,
        "description",
        &format!("{} scenes, {} total detections", results.len(), results.iter().map(|r| r.detections.len()).sum::<usize>()),
    )?;

    // Styles for each method
    for method in &[
        DetectionMethod::DarkSpot,
        DetectionMethod::ClearHole,
        DetectionMethod::SedimentTrap,
    ] {
        write_style(&mut xml, method)?;
    }

    // Group results by lake
    let mut by_lake: std::collections::HashMap<String, Vec<&DetectionResult>> =
        std::collections::HashMap::new();
    for r in results {
        by_lake
            .entry(format!("{}", r.lake))
            .or_default()
            .push(r);
    }

    for (lake_name, lake_results) in &by_lake {
        // <Folder> per lake
        xml.write_event(Event::Start(BytesStart::new("Folder")))?;
        write_text_element(&mut xml, "name", lake_name)?;

        for result in lake_results {
            // <Folder> per scene
            xml.write_event(Event::Start(BytesStart::new("Folder")))?;
            let scene_label = format!(
                "{} — {} [{}]",
                result.scene_id,
                result.method.label(),
                &result.datetime,
            );
            write_text_element(&mut xml, "name", &scene_label)?;

            // Weather info as description
            if let Some(ref wx) = result.weather {
                write_text_element(
                    &mut xml,
                    "description",
                    &format!(
                        "Weather: {} (confidence {:.0}%)\n{}",
                        wx.window
                            .map(|w| format!("{:?}", w))
                            .unwrap_or_else(|| "none".into()),
                        wx.confidence * 100.0,
                        wx.reason
                    ),
                )?;
            }

            // Placemarks for each detection
            for (i, det) in result.detections.iter().enumerate() {
                write_placemark(&mut xml, det, &result.method, i + 1)?;
            }

            xml.write_event(Event::End(BytesEnd::new("Folder")))?; // scene
        }

        xml.write_event(Event::End(BytesEnd::new("Folder")))?; // lake
    }

    xml.write_event(Event::End(BytesEnd::new("Document")))?;
    xml.write_event(Event::End(BytesEnd::new("kml")))?;

    Ok(())
}

fn write_style<W: Write>(xml: &mut Writer<W>, method: &DetectionMethod) -> Result<()> {
    let mut style = BytesStart::new("Style");
    style.push_attribute(("id", method.label()));
    xml.write_event(Event::Start(style))?;

    // IconStyle
    xml.write_event(Event::Start(BytesStart::new("IconStyle")))?;
    write_text_element(xml, "color", method_color(*method))?;
    write_text_element(xml, "scale", "1.2")?;
    xml.write_event(Event::Start(BytesStart::new("Icon")))?;
    write_text_element(xml, "href", method_icon(*method))?;
    xml.write_event(Event::End(BytesEnd::new("Icon")))?;
    xml.write_event(Event::End(BytesEnd::new("IconStyle")))?;

    // LabelStyle
    xml.write_event(Event::Start(BytesStart::new("LabelStyle")))?;
    write_text_element(xml, "color", method_color(*method))?;
    write_text_element(xml, "scale", "0.8")?;
    xml.write_event(Event::End(BytesEnd::new("LabelStyle")))?;

    xml.write_event(Event::End(BytesEnd::new("Style")))?;
    Ok(())
}

fn write_placemark<W: Write>(
    xml: &mut Writer<W>,
    det: &Detection,
    method: &DetectionMethod,
    idx: usize,
) -> Result<()> {
    xml.write_event(Event::Start(BytesStart::new("Placemark")))?;

    let name = format!(
        "#{} {} σ={:.1} conf={:.0}%",
        idx,
        det.classification,
        det.anomaly_sigma,
        det.confidence * 100.0,
    );
    write_text_element(xml, "name", &name)?;

    let desc = format!(
        "Lat: {:.6}°  Lon: {:.6}°\nRadius: {:.0} m\nAnomaly: {:.1}σ\nConfidence: {:.1}%\nMethod: {}",
        det.lat, det.lon, det.radius_m, det.anomaly_sigma, det.confidence * 100.0, method.label()
    );
    write_text_element(xml, "description", &desc)?;

    write_text_element(xml, "styleUrl", &format!("#{}", method.label()))?;

    // Point
    xml.write_event(Event::Start(BytesStart::new("Point")))?;
    write_text_element(
        xml,
        "coordinates",
        &format!("{:.6},{:.6},0", det.lon, det.lat),
    )?;
    xml.write_event(Event::End(BytesEnd::new("Point")))?;

    xml.write_event(Event::End(BytesEnd::new("Placemark")))?;
    Ok(())
}

/// Write a simple text element like `<name>value</name>`.
fn write_text_element<W: Write>(xml: &mut Writer<W>, tag: &str, text: &str) -> Result<()> {
    xml.write_event(Event::Start(BytesStart::new(tag)))?;
    xml.write_event(Event::Text(BytesText::new(text)))?;
    xml.write_event(Event::End(BytesEnd::new(tag)))?;
    Ok(())
}

/// Generate a search-area polygon KML for a lake's bounds.
pub fn lake_search_area_kml(lake: Lake, bounds: [f64; 4]) -> Result<String> {
    let mut buf = Vec::new();
    let mut xml = Writer::new_with_indent(&mut buf, b' ', 2);

    xml.write_event(Event::Decl(BytesDecl::new("1.0", Some("UTF-8"), None)))?;
    let mut kml = BytesStart::new("kml");
    kml.push_attribute(("xmlns", "http://www.opengis.net/kml/2.2"));
    xml.write_event(Event::Start(kml))?;
    xml.write_event(Event::Start(BytesStart::new("Document")))?;
    write_text_element(&mut xml, "name", &format!("{} Search Area", lake))?;

    xml.write_event(Event::Start(BytesStart::new("Placemark")))?;
    write_text_element(&mut xml, "name", &format!("{} Bounds", lake))?;

    xml.write_event(Event::Start(BytesStart::new("Polygon")))?;
    xml.write_event(Event::Start(BytesStart::new("outerBoundaryIs")))?;
    xml.write_event(Event::Start(BytesStart::new("LinearRing")))?;

    // bounds = [west_lon, south_lat, east_lon, north_lat]
    let coords = format!(
        "{lon_w},{lat_s},0 {lon_e},{lat_s},0 {lon_e},{lat_n},0 {lon_w},{lat_n},0 {lon_w},{lat_s},0",
        lon_w = bounds[0],
        lat_s = bounds[1],
        lon_e = bounds[2],
        lat_n = bounds[3],
    );
    write_text_element(&mut xml, "coordinates", &coords)?;

    xml.write_event(Event::End(BytesEnd::new("LinearRing")))?;
    xml.write_event(Event::End(BytesEnd::new("outerBoundaryIs")))?;
    xml.write_event(Event::End(BytesEnd::new("Polygon")))?;
    xml.write_event(Event::End(BytesEnd::new("Placemark")))?;
    xml.write_event(Event::End(BytesEnd::new("Document")))?;
    xml.write_event(Event::End(BytesEnd::new("kml")))?;

    Ok(String::from_utf8(buf)?)
}
