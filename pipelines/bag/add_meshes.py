import os

with open('src/lib.rs', 'a', encoding='utf-8') as f:
    f.write('''

#[pyfunction]
fn generate_3d_meshes(
    uncertainty_data: Vec<f32>,
    rows: usize,
    cols: usize,
    cell_size_m: f64,
    output_dir: String,
    bag_name: String,
) -> PyResult<usize> {
    use std::path::Path;
    let out_dir = Path::new(&output_dir);
    if !out_dir.exists() {
        std::fs::create_dir_all(out_dir).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    }
    
    // Attempt reconstruction from Python data
    let array_res = ndarray::Array2::from_shape_vec((rows, cols), uncertainty_data);
    let uncert_array = match array_res {
        Ok(arr) => arr,
        Err(e) => return Err(pyo3::exceptions::PyValueError::new_err(format!("Shape mismatch: {}", e))),
    };

    let params = crate::bag_mesh::SonarParams::default();
    
    let objects = crate::bag_mesh::detect_masked_regions(&uncert_array, cell_size_m);
    
    let mut exported = 0;
    for (i, obj) in objects.iter().enumerate() {
        let (mesh, _) = crate::bag_mesh::build_mesh(obj, &params);
        if mesh.vertices.is_empty() {
            continue;
        }

        let obj_name = format!("{}_{}.obj", bag_name, i);
        let obj_path = out_dir.join(&obj_name);
        if let Err(e) = crate::bag_mesh::write_obj(&mesh, &obj_path) {
            eprintln!("Failed to write OBJ: {}", e);
        } else {
            exported += 1;
        }
    }
    Ok(exported)
}
''')

with open('src/lib.rs', 'r', encoding='utf-8') as f:
    text = f.read()

text = text.replace('m.add_function(wrap_pyfunction!(find_references_in_extent, m)?)?;', 'm.add_function(wrap_pyfunction!(find_references_in_extent, m)?)?;\n    m.add_function(wrap_pyfunction!(generate_3d_meshes, m)?)?;')

with open('src/lib.rs', 'w', encoding='utf-8') as f:
    f.write(text)
