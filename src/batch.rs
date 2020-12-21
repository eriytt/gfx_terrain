
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
    pub fn update(&self) -> Vec<u32> {
        vec![]
    }

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
