#[macro_use]
extern crate gfx;
extern crate image;
extern crate itertools;

use gfx::traits::FactoryExt;
use image::GenericImageView;
use itertools::Itertools;

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
    slice: gfx::Slice<R>,
}

impl <R: gfx::Resources> GraphicsData <R> {
    fn new<F: FactoryExt<R>>(factory: &mut F,
                             rt: gfx::handle::RawRenderTargetView<R>,
                             primitive: gfx::Primitive)
                             -> GraphicsData<R> {
        // let pso = factory.create_pipeline_simple(&VERTEX_SHADER, &PIXEL_SHADER, pipe::new())
        //     .expect("Error creating pipeline state:");

        let pso = GraphicsData::<R>::create_pipeline(factory, primitive);

        let (vertex_buffer, slice) = factory.create_vertex_buffer_with_slice(&[], ());
        let mvp_buffer = factory.create_constant_buffer(1);
        use gfx::memory::Typed;
        GraphicsData {
            pso,
            pipe_data: pipe::Data {
                vbuf: vertex_buffer,
                out_color: Typed::new(rt),
                transform: mvp_buffer,
            },
            slice
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

    // fn create_pipeline_simple<I: pso::PipelineInit>(&mut self, factory: I, vs: &[u8], ps: &[u8], init: I)
    //                                                 -> Result<pso::PipelineState<R, I::Meta>,
    //                                                           PipelineStateError<String>>
    // {
    //     let set = try!(self.create_shader_set(vs, ps));
    //     self.create_pipeline_state(&set, Primitive::TriangleList, state::Rasterizer::new_fill(),
    //                                init)
    // }

}


#[allow(dead_code)]
pub struct Terrain <R: gfx::Resources, F: FactoryExt<R>> {
    terrain_data: Vec<Vec<f32>>,
    grid_distance: f32,
    view_mat: [[f32; 4]; 4],
    projection_mat: [[f32; 4]; 4],
    needs_transform_update: bool,
    gfx_data: Option<GraphicsData<R>>,
    needs_tesselation: bool,
    draw_wireframe: bool,
    factory: F
}


#[allow(dead_code)]
impl <R: gfx::Resources, F: FactoryExt<R>> Terrain <R, F>{
    pub fn from_data(data: Vec<Vec<f32>>, size: u16,  factory: F) -> Result<Self, String> {
        if !u16::is_power_of_two(size) {
            return Result::Err(String::from("size must be power of two"));
        }

        for (i, v) in data.iter().enumerate() {
            if v.len() as u16 != size + 1{
                return Result::Err(String::from(format!(
                    "data[{}] does not match size, {} != {}", i, v.len(), size)))
            }
        }
        if data.len() as u16 != size + 1 {
            Result::Err(String::from(format!("data does not match size, {} != {}", data.len(), size)))
        } else {
            Ok(Terrain {
                terrain_data: data,
                gfx_data: None,
                view_mat: Default::default(),
                projection_mat: Default::default(),
                grid_distance: 1.0,
                needs_transform_update: true,
                needs_tesselation: true,
                draw_wireframe: false,
                factory: factory,
            })
        }
    }

    pub fn from_file(filename: &str, scale: f32, factory: F) -> Result<Self, String> {
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

        Terrain::from_data(v, (width - 1)  as u16, factory)
    }

    pub fn  set_view_matrix(&mut self, view_matrix: &[[f32; 4]; 4]) {
        self.view_mat = view_matrix.clone();
        self.needs_transform_update = true;
    }

    pub fn  set_projection_matrix(&mut self, projection_matrix: &[[f32; 4]; 4]) {
        self.projection_mat = projection_matrix.clone();
        self.needs_transform_update = true;
    }

    fn needs_retesselation(&self) -> bool {
        self.needs_tesselation
    }

    pub fn draw<C: gfx::CommandBuffer<R>>(
        &mut self,
        encoder: &mut gfx::Encoder<R, C>,
        rt: gfx::handle::RawRenderTargetView<R>,
    ) {
        if self.gfx_data.is_none() {
            self.gfx_data = Some(GraphicsData::new(&mut self.factory, rt, gfx::Primitive::TriangleStrip));
        }
        //let gfx_data = self.gfx_data.unwrap_or_else(|| GraphicsData::new(factory, rt));

        if self.needs_transform_update {
            let gfx_data = self.gfx_data.as_mut().unwrap();
            let mvp_transform = Transform {
                view: self.view_mat,
                projection: self.projection_mat,
            };
            encoder.update_buffer(&gfx_data.pipe_data.transform, &[mvp_transform], 0)
                .expect("Failed to update transform:");
        }
        if self.needs_retesselation() {
            //let (vertices, indices) = self.update_wireframe();
            let (vertices, indices) = self.update();
            let gfx_data = self.gfx_data.as_mut().unwrap();
            let index_buffer = self.factory.create_index_buffer(indices.as_slice());
            let (vertex_buffer, slice) = self.factory.create_vertex_buffer_with_slice(&vertices, index_buffer);
            gfx_data.pipe_data.vbuf = vertex_buffer;
            gfx_data.slice = slice;
            self.needs_tesselation = false;
        }

        let gfx_data = self.gfx_data.as_ref().unwrap();
        encoder.draw(&gfx_data.slice, &gfx_data.pso, &gfx_data.pipe_data);
    }

    pub fn toggle_wireframe(&mut self) {
        if self.draw_wireframe {
            self.draw_wireframe = false;
            self.needs_tesselation = true;
            let mut gfx_data = self.gfx_data.as_mut().unwrap();
            gfx_data.pso = GraphicsData::<R>::create_pipeline(&mut self.factory, gfx::Primitive::TriangleStrip);
        } else {
            self.draw_wireframe = true;
            self.needs_tesselation = true;
            let mut gfx_data = self.gfx_data.as_mut().unwrap();
            gfx_data.pso = GraphicsData::<R>::create_pipeline(&mut self.factory, gfx::Primitive::LineStrip);
        };
    }
}

const WHITE: [f32; 3] = [1.0, 1.0, 1.0];
const BLUE: [f32; 3] = [0.0, 0.0, 1.0];

#[allow(dead_code)]
impl <R: gfx::Resources, F: FactoryExt<R>> Terrain <R, F>{

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
        //let z_step_back_first  = |row: u32| vec![index_of(0, row)].into_iter();

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
