mod ffi {
    extern "C" {
        pub fn mat_mat(rows: usize, cols: usize) -> *mut Mat;
        pub fn mat_from(rows: usize, cols: usize, values: *const f32) -> *mut Mat;
        pub fn mat_eye(dim: usize) -> *mut Mat;
        pub fn mat_rmul(a: *const Mat, b: *const Mat) -> *mut Mat;
        pub fn mat_radd(a: *const Mat, b: *const Mat) -> *mut Mat;
        pub fn mat_at(m: *const Mat, row: usize, col: usize) -> f32;
        pub fn mat_equals(a: *const Mat, b: *const Mat) -> bool;
        pub fn mat_print(m: *const Mat);
        pub fn mat_free_mat(m: *mut Mat);
    }

    #[repr(C)]
    pub struct Mat {
        pub rows: usize,
        pub cols: usize,
        pub data: *mut f32,
    }
}

pub struct Mat(*mut ffi::Mat);

impl Mat {
    pub fn new(rows: usize, cols: usize) -> Self {
        Mat(unsafe { ffi::mat_mat(rows, cols) })
    }

    pub fn from_slice(rows: usize, cols: usize, data: &[f32]) -> Self {
        Mat(unsafe { ffi::mat_from(rows, cols, data.as_ptr()) })
    }

    pub fn eye(dim: usize) -> Self {
        Mat(unsafe { ffi::mat_eye(dim) })
    }

    pub fn mul(&self, other: &Mat) -> Self {
        Mat(unsafe { ffi::mat_rmul(self.0, other.0) })
    }

    pub fn add(&self, other: &Mat) -> Self {
        Mat(unsafe { ffi::mat_radd(self.0, other.0) })
    }

    pub fn at(&self, row: usize, col: usize) -> f32 {
        unsafe { ffi::mat_at(self.0, row, col) }
    }

    pub fn equals(&self, other: &Mat) -> bool {
        unsafe { ffi::mat_equals(self.0, other.0) }
    }

    pub fn print(&self) {
        unsafe { ffi::mat_print(self.0) }
    }
}

impl Drop for Mat {
    fn drop(&mut self) {
        unsafe { ffi::mat_free_mat(self.0) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eye() {
        let m = Mat::eye(3);
        assert_eq!(m.at(0, 0), 1.0);
        assert_eq!(m.at(1, 1), 1.0);
        assert_eq!(m.at(2, 2), 1.0);
        assert_eq!(m.at(0, 1), 0.0);
    }

    #[test]
    fn test_mul() {
        let a = Mat::from_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let b = Mat::eye(2);
        let c = a.mul(&b);
        assert!(c.equals(&a));
    }
}
