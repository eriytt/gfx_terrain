
pub struct Batch {
    pub x_idx: u16,
    pub y_idx: u16,
    array_size: u16,
    pub batch_size: u16,
}

impl Batch {
    pub fn new(x_idx: u16, y_idx: u16, array_size: u16, batch_size: u16) -> Self {
        Self {
            x_idx,
            y_idx,
            array_size,
            batch_size,
        }
    }

    #[allow(dead_code)]
    pub fn update(&self, lod: u32) -> Vec<u32> {
        let lod: u32 = 2u32.pow(lod);
        let d_size = self.array_size as u32;
        let a_size = self.batch_size as u32 / lod;

        let index_of = |x: u32, y: u32| y * d_size + x;

        let z_step_last  = |row: u32| vec![index_of(a_size, row + 1)].into_iter();
        let z_step_back_last  = |row: u32| vec![index_of(a_size, row)].into_iter();

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

        let skirt_vertex_idx = d_size * d_size;
        let skirt_y_max = || (1..a_size).flat_map(
            |i| vec![skirt_vertex_idx, index_of(i, a_size), index_of(i + 1, a_size)].into_iter());
        let skirt_x_max = || (0..a_size).rev().flat_map(
            |i| vec![skirt_vertex_idx, index_of(a_size, i + 1), index_of(a_size, i)].into_iter());
        let skirt_y_min = || (0..a_size).rev().flat_map(
            |i| vec![skirt_vertex_idx, index_of(i + 1, 0), index_of(i, 0)].into_iter());
        let skirt_x_min = || (0..a_size).flat_map(
            |i| vec![skirt_vertex_idx, index_of(0, i), index_of(0, i + 1)].into_iter());

        let batch_offset = (self.y_idx as u32 * self.batch_size as u32 * self.array_size as u32)
            + self.batch_size as u32 * self.x_idx as u32;

        (0..a_size).flat_map(|idx| select_row(idx))
            .chain(skirt_y_max())
            .chain(skirt_x_max())
            .chain(skirt_y_min())
            .chain(skirt_x_min())
            .map(|i| { if i == skirt_vertex_idx { skirt_vertex_idx } else { i * lod + batch_offset } })
            .collect()
    }

    #[allow(dead_code)]
    pub fn update_wireframe(&self, lod: u32) -> Vec<u32> {
        let lod: u32 = 2u32.pow(lod);
        let d_size = self.array_size as u32;
        let a_size = self.batch_size as u32 / lod;

        let index_of = |x: u32, y: u32| y * d_size + x;

        let row_line = |row| {(0..a_size).map(move |r| index_of(r + 1, row))};

        let z_step_last  = |row: u32| vec![index_of(a_size, row + 1)].into_iter();

        let zig_zag = |col, row| {vec![
            index_of(col, row),
            index_of(col, row + 1),
        ]};


        let row = |row| {
            row_line(row).into_iter()
                .chain(z_step_last(row))
                .chain(((0..a_size).rev()).flat_map(move |col| zig_zag(col, row)))
        };

        let batch_offset = (self.y_idx as u32 * self.batch_size as u32 * self.array_size as u32)
            + self.batch_size as u32 * self.x_idx as u32;

        (0..=1)
            .chain((0..a_size).flat_map(move |idx| row(idx)))
            .chain(row_line(a_size))
            .map(|i| i * lod + batch_offset) // TODO fix mapping
            .collect()
    }

}
