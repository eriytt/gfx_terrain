#[macro_use]
extern crate gfx;
extern crate image;
extern crate itertools;
extern crate vecmath;


use gfx::traits::FactoryExt;
use image::GenericImageView;
use itertools::Itertools;
use vecmath::{ Vector3, Vector4, Matrix4, vec4_dot, col_mat4_mul, mat4_inv};

const VERTEX_SHADER: &'static [u8] = b"
#version 150 core

in vec4 a_Pos;
in vec3 a_Color;

uniform Transform {
   mat4 u_View;
   mat4 u_Projection;
};

out vec4 v_Color;

void main() {
    v_Color = vec4(a_Color, 1.0);
    gl_Position = u_Projection * u_View * a_Pos;
}
";

const PIXEL_SHADER: &'static [u8] =  b"
#version 150 core

in vec4 v_Color;
out vec4 Target0;

void main() {
    Target0 = v_Color;
}
";


type ColorFormat = gfx::format::Srgba8;

gfx_defines!{
    vertex Vertex {
        pos: [f32; 4] = "a_Pos",
        color: [f32; 3] = "a_Color",
    }

    constant Transform {
        view: [[f32; 4]; 4] = "u_View",
        projection: [[f32; 4]; 4] = "u_Projection",
    }

    pipeline pipe {
        vbuf: gfx::VertexBuffer<Vertex> = (),
        transform: gfx::ConstantBuffer<Transform> = "Transform",
        out_color: gfx::RenderTarget<ColorFormat> = "Target0",
    }
}


struct GraphicsData <R: gfx::Resources>{
    pso: gfx::PipelineState<R, pipe::Meta>,
    pipe_data: pipe::Data<R>,
}

impl <R: gfx::Resources> GraphicsData <R> {
    fn new<F: FactoryExt<R>>(factory: &mut F,
                             rt: gfx::handle::RawRenderTargetView<R>,
                             primitive: gfx::Primitive)
                             -> GraphicsData<R> {
        let pso = GraphicsData::<R>::create_pipeline(factory, primitive);

        let vertex_buffer = factory.create_vertex_buffer(&[]);
        let mvp_buffer = factory.create_constant_buffer(1);
        use gfx::memory::Typed;
        GraphicsData {
            pso,
            pipe_data: pipe::Data {
                vbuf: vertex_buffer,
                out_color: Typed::new(rt),
                transform: mvp_buffer,
            },
        }
    }


    fn create_pipeline<F: FactoryExt<R>>
        (factory: &mut F,
        primitive: gfx::Primitive)
         -> gfx::PipelineState<R, pipe::Meta> {
            let set = factory.create_shader_set(&VERTEX_SHADER, &PIXEL_SHADER).expect("Failed to create shader set");
            factory.create_pipeline_state(&set, primitive, gfx::state::Rasterizer::new_fill(), pipe::new())
                .expect("Error creating pipeline state")
        }
}

mod batch;

struct TBatch <R: gfx::Resources> {
    batch: batch::Batch,
    slice: gfx::Slice<R>,
    current_lod: u32,
    min: f32,
    max: f32
}

impl <R: gfx::Resources> TBatch <R> {
    fn bound(&self) -> Bound {
        let min = [
            (self.batch.x_idx * self.batch.batch_size) as f32,
            self.min,
            (self.batch.y_idx * self.batch.batch_size) as f32
        ];
        let max = [
            ((self.batch.x_idx + 1) * self.batch.batch_size) as f32,
            self.max,
            ((self.batch.y_idx + 1) * self.batch.batch_size) as f32
        ];

        Bound {
            min,
            max
        }
    }
}

#[allow(dead_code)]
pub struct Terrain <R: gfx::Resources, F: FactoryExt<R>> {
    terrain_data: Vec<Vec<f32>>,
    grid_distance: f32,
    view_mat: [[f32; 4]; 4],
    cull_matrix: Option<Matrix4<f32>>,
    max_lod: u32,
    projection_mat: [[f32; 4]; 4],
    needs_transform_update: bool,
    gfx_data: Option<GraphicsData<R>>,
    needs_tesselation: bool,
    draw_wireframe: bool,
    factory: F,
    batches: Vec<TBatch<R>>
}


#[allow(dead_code)]
impl <R: gfx::Resources, F: FactoryExt<R>> Terrain <R, F>{
    pub fn from_data(data: Vec<Vec<f32>>, size: u16, batch_size: u16, factory: F) -> Result<Self, String> {
        if !u16::is_power_of_two(size) {
            return Result::Err(String::from("Size must be power of two"));
        }

        for (i, v) in data.iter().enumerate() {
            if v.len() as u16 != size + 1{
                return Result::Err(String::from(format!(
                    "data[{}] does not match size, {} != {}", i, v.len(), size)))
            }
        }

        if !u16::is_power_of_two(batch_size) {
            return Result::Err(String::from("Batch size must be power of two"));
        }

        if batch_size > size {
            return Result::Err(String::from("Batch size must be lesser or equal to size"));
        }

        if data.len() as u16 != size + 1 {
            Result::Err(String::from(format!("data does not match size, {} != {}", data.len(), size)))
        } else {
            let num_batches = size / batch_size as u16;
            let batches = (0..(num_batches * num_batches))
                .map(|i| {
                    let (min, max) = Self::batch_vertical_bounds(
                        &data,
                        (i % num_batches) * batch_size,
                        i / num_batches * batch_size,
                        batch_size
                    );
                    TBatch {
                        batch: batch::Batch::new(i % num_batches, i / num_batches, size + 1, batch_size),
                        slice: gfx::Slice::from_vertex_count(0),
                        current_lod: u32::MAX,
                        min,
                        max
                    }
                })
                .collect();

            Ok(Terrain {
                terrain_data: data,
                gfx_data: None,
                view_mat: Default::default(),
                cull_matrix: None,
                max_lod: 16 - (batch_size as u16).leading_zeros() - 1,
                projection_mat: Default::default(),
                grid_distance: 1.0,
                needs_transform_update: true,
                needs_tesselation: true,
                draw_wireframe: true,
                factory,
                batches
            })
        }
    }

    pub fn from_file(filename: &str, scale: f32, batch_size: u16, factory: F) -> Result<Self, String> {
        let img = match image::open(filename) {
            Ok(i) => i,
            Err(e) => return Err(String::from(format!("Failed to load image: {:}", e)))
        };

        let (width, height) = img.dimensions();

        if width > u16::MAX.into() || height > u16::MAX.into() {
            return Result::Err(String::from(
                format!("Image dimensions exceed u16 size: {}x{}", width, height)));
        }

        let mut v = Vec::<Vec<f32>>::with_capacity(width as usize);
        use image::Pixel;
        for row in img.pixels().chunks(width as usize).into_iter() {
             v.push(row.map(|(_x, _y, p)| {p.to_luma().channels()[0] as f32 * scale}).collect());
        }

        Terrain::from_data(v, (width - 1)  as u16, batch_size, factory)
    }

    pub fn  set_view_matrix(&mut self, view_matrix: &[[f32; 4]; 4]) {
        self.view_mat = view_matrix.clone();
        self.needs_transform_update = true;
    }

    pub fn set_cull_view_matrix(&mut self, cull_matrix: Option<Matrix4<f32>>) {
        self.cull_matrix = cull_matrix;
    }

    pub fn  set_projection_matrix(&mut self, projection_matrix: &[[f32; 4]; 4]) {
        self.projection_mat = projection_matrix.clone();
        self.needs_transform_update = true;
    }

    fn needs_retesselation(&self) -> bool {
        self.needs_tesselation
    }

    pub fn draw<C: gfx::CommandBuffer<R>, LF: Fn(f32, f32, f32) -> u32>(
        &mut self,
        encoder: &mut gfx::Encoder<R, C>,
        rt: gfx::handle::RawRenderTargetView<R>,
        lod_fn: LF,
    ) {
        if self.gfx_data.is_none() {
            let primitive = if self.draw_wireframe {
                gfx::Primitive::LineStrip
            } else {
                gfx::Primitive::TriangleStrip
            };
            self.gfx_data = Some(GraphicsData::new(&mut self.factory, rt, primitive));
        }

        if self.needs_transform_update {
            let gfx_data = self.gfx_data.as_mut().unwrap();
            let mvp_transform = Transform {
                view: self.view_mat,
                projection: self.projection_mat,
            };
            encoder.update_buffer(&gfx_data.pipe_data.transform, &[mvp_transform], 0)
                .expect("Failed to update transform:");
        }
        let needs_retesselation = self.needs_retesselation();

        let cull_view_matrix = self.cull_matrix.unwrap_or(self.view_mat);
        let pos = mat4_inv(cull_view_matrix)[3];
        let vp = col_mat4_mul(self.projection_mat, cull_view_matrix);
        let frustum = Self::extract_planes_from_projmat(vp);

        let draw_batches = self.cull_batches(&frustum, &pos);
        for (tb, draw) in self.batches.iter_mut().zip(draw_batches.iter()) {
            if !draw { continue; }

            let b = &tb.batch;

            let lod = lod_fn((b.x_idx * b.batch_size) as f32 * self.grid_distance,
                             (b.y_idx * b.batch_size) as f32 * self.grid_distance,
                             self.grid_distance * b.batch_size as f32,
            );

            let lod = std::cmp::min(lod, self.max_lod);

            if lod == tb.current_lod && !needs_retesselation {
                continue;
            }

            tb.current_lod = lod;
            let indices = b.update_wireframe(tb.current_lod);
            let index_buffer = self.factory.create_index_buffer(indices.as_slice());
            tb.slice = gfx::Slice {
                start: 0,
                end: indices.len() as u32,
                base_vertex: 0,
                instances: None,
                buffer: index_buffer
            }
        }

       if needs_retesselation {
           let vertices = self.update_vertex_array();
           let vertex_buffer = self.factory.create_vertex_buffer(&vertices);
           let gfx_data = self.gfx_data.as_mut().unwrap();
           gfx_data.pipe_data.vbuf = vertex_buffer;
           self.needs_tesselation = false;
       }

        let gfx_data = self.gfx_data.as_ref().unwrap();
        for (b, draw) in self.batches.iter_mut().zip(draw_batches.iter()) {
            if !draw { continue; }
            encoder.draw(&b.slice, &gfx_data.pso, &gfx_data.pipe_data);
        }
    }

    pub fn toggle_wireframe(&mut self) {
        if self.draw_wireframe {
            self.draw_wireframe = false;
            self.needs_tesselation = true;
            let mut gfx_data = self.gfx_data.as_mut().unwrap();
            gfx_data.pso = GraphicsData::<R>::create_pipeline(&mut self.factory,
                                                              gfx::Primitive::TriangleStrip);
        } else {
            self.draw_wireframe = true;
            self.needs_tesselation = true;
            let mut gfx_data = self.gfx_data.as_mut().unwrap();
            gfx_data.pso = GraphicsData::<R>::create_pipeline(&mut self.factory,
                                                              gfx::Primitive::LineStrip);
        };
    }
}

const WHITE: [f32; 3] = [1.0, 1.0, 1.0];
const BLUE: [f32; 3] = [0.0, 0.0, 1.0];

type Vector3f = Vector3<f32>;
type Vector4f = Vector4<f32>;
type Frustum = [Vector4f; 6];

struct Bound {
    min: Vector3f,
    max: Vector3f
}

#[allow(dead_code)]
impl <R: gfx::Resources, F: FactoryExt<R>> Terrain <R, F>{

    fn batch_vertical_bounds(data: &Vec<Vec<f32>>,
                             x_start: u16, y_start: u16,
                             batch_size: u16) -> (f32, f32) {
        let mut min = f32::MAX;
        let mut max = 0.0f32;
        for y in y_start..(y_start + batch_size + 1) {
            for x in x_start..(x_start  + batch_size + 1) {
                min = min.min(data[x as usize][y as usize]);
                max = max.max(data[x as usize][y as usize]);
            }
        }
        (min, max)
    }

    fn camera_inside_batch(&self, pos: &Vector4f, batch: &TBatch<R>) -> bool {
        let bound = batch.bound();
        pos[0] >= bound.min[0] && pos[0] < bound.max[0]
            && pos[2] >= bound.min[2] && pos[2] < bound.max[2]
    }

    fn cull_batches(&self, frustum: &Frustum, position: &Vector4f) -> Vec<bool> {
        self.batches.iter().map(|b| {
            self.camera_inside_batch(position, &b)
                || !Self::box_fully_outside_frustum(&frustum, &b.bound())
        }).collect()
    }

    fn extract_planes_from_projmat(mat: Matrix4<f32>) -> Frustum {
        let left: Vec<f32>   = (0..4).map(|i| { mat[i][3] + mat[i][0] }).collect();
        let right: Vec<f32>  = (0..4).map(|i| { mat[i][3] - mat[i][0] }).collect();
        let bottom: Vec<f32> = (0..4).map(|i| { mat[i][3] + mat[i][1] }).collect();
        let top: Vec<f32>    = (0..4).map(|i| { mat[i][3] - mat[i][1] }).collect();
        let near: Vec<f32>   = (0..4).map(|i| { mat[i][3] + mat[i][2] }).collect();
        let far: Vec<f32>    = (0..4).map(|i| { mat[i][3] - mat[i][2] }).collect();

        fn vec_to_array(v: Vec<f32>) -> Vector4f {
            [v[0], v[1], v[2], v[3]]
        }

        [
            vec_to_array(left),
            vec_to_array(right),
            vec_to_array(bottom),
            vec_to_array(top),
            vec_to_array(near),
            vec_to_array(far)
        ]

        // Should work when it's possible to move to 1.48.0
        // use std::convert::TryInto;
        // [
        //     left.try_into().unwrap(),
        //     right.try_into().unwrap(),
        //     top.try_into().unwrap(),
        //     bottom.try_into().unwrap(),
        //     near.try_into().unwrap(),
        //     far.try_into().unwrap()
        // ]
    }

    fn box_fully_outside_frustum( fru: &Frustum, bound: &Bound) -> bool
    {
        let point_inside_frustum = |point: Vector4f| {
            for i in 0..6 {
                if vec4_dot( fru[i], point) < 0.0 { return false; }
            }
            true
        };

        if point_inside_frustum([bound.min[0], bound.min[1], bound.min[2], 1.0]) { return false; }
        if point_inside_frustum([bound.max[0], bound.min[1], bound.min[2], 1.0]) { return false; }
        if point_inside_frustum([bound.min[0], bound.max[1], bound.min[2], 1.0]) { return false; }
        if point_inside_frustum([bound.max[0], bound.max[1], bound.min[2], 1.0]) { return false; }
        if point_inside_frustum([bound.min[0], bound.min[1], bound.max[2], 1.0]) { return false; }
        if point_inside_frustum([bound.max[0], bound.min[1], bound.max[2], 1.0]) { return false; }
        if point_inside_frustum([bound.min[0], bound.max[1], bound.max[2], 1.0]) { return false; }
        if point_inside_frustum([bound.max[0], bound.max[1], bound.max[2], 1.0]) { return false; }

        true
}

    fn update_vertex_array(&self) -> Vec<Vertex> {
        let a_size = self.terrain_data.len();
        let mut v: Vec<Vertex> = Vec::with_capacity(a_size * a_size);

        for y in 0..a_size {
            for x in 0..a_size {
                v.push(Vertex { pos: [ x as f32 * self.grid_distance,
                                       self.terrain_data[x][y],
                                       y as f32 * self.grid_distance,
                                       1.0],
                                color: if (x % 2) == 0 || (y % 2) == 0 {WHITE} else {BLUE}});
            }
        }
        v
    }

    fn update(&self) -> (Vec<Vertex>, Vec<u32>) {

        let d_size = self.terrain_data.len() as u32;
        let a_size = d_size - 1;
        let index_of = |x: u32, y: u32| y * d_size + x;

        let z_step_last  = |row: u32| vec![index_of(d_size - 1, row + 1)].into_iter();
        let z_step_back_last  = |row: u32| vec![index_of(d_size - 1, row)].into_iter();

        let tri_even = |col, row| vec![index_of(col, row), index_of(col + 1, row +1)].into_iter();
        let tri_odd  = |col, row| vec![index_of(col, row), index_of(col, row + 1)].into_iter();

        let even_row = |row| {
            std::iter::once(index_of(0, row + 1))
                .chain((0..a_size).flat_map(move |idx| tri_even(idx, row)))
                .chain(z_step_back_last(row))
                .chain(z_step_last(row))
        };

        let odd_row = |row| {
            z_step_last(row)
                .chain((0..a_size).rev().flat_map(move |idx|tri_odd(idx, row)))
        };

        let select_row =  |idx: u32|{
            if idx & 1 == 1 {
                Box::new(odd_row(idx)) as Box<dyn Iterator<Item = u32>>
            } else {
                Box::new(even_row(idx)) as Box<dyn Iterator<Item = u32>>
            }
        };


        let tri_indices: Vec<u32> =
            (0..d_size).flat_map(|idx| select_row(idx))
            .collect();

        (self.update_vertex_array(), tri_indices)

    }

    fn update_wireframe(&self) -> (Vec<Vertex>, Vec<u32>) {

        let d_size = self.terrain_data.len() as u32;
        let a_size = d_size - 1;

        let index_of = |x: u32, y: u32| y * d_size + x;

        let row_line = |row| {(0..(d_size - 1)).map(move |r| index_of(r + 1, row))};

        let z_step_last  = |row: u32| vec![index_of(d_size - 1, row + 1)].into_iter();

        let zig_zag = |col, row| {vec![
            index_of(col, row),
            index_of(col, row + 1),
        ]};


        let row = |row| {
            row_line(row).into_iter()
                .chain(z_step_last(row))
                .chain(((0..a_size).rev()).flat_map(move |col| zig_zag(col, row)))
        };

        let tri_indices: Vec<u32> =
            (0..=1)
            .chain((0..d_size).flat_map(move |idx| row(idx)))
            .chain(row_line(a_size))
            .collect();

        (self.update_vertex_array(), tri_indices)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
