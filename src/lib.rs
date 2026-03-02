mod ffi {
    extern "C" {
        pub fn mat_mat(rows: usize, cols: usize) -> *mut Mat;
        pub fn mat_from(rows: usize, cols: usize, values: *const f32) -> *mut Mat;
        pub fn mat_reye(dim: usize) -> *mut Mat;
        pub fn mat_rmul(a: *const Mat, b: *const Mat) -> *mut Mat;
        pub fn mat_radd(a: *const Mat, b: *const Mat) -> *mut Mat;
        pub fn mat_at(m: *const Mat, row: usize, col: usize) -> f32;
        pub fn mat_equals(a: *const Mat, b: *const Mat) -> bool;
        pub fn mat_print(m: *const Mat);
        pub fn mat_free_mat(m: *mut Mat);
        pub fn mat_vec(dim: usize) -> *mut Mat;
        pub fn mat_eigvals(out: *mut Mat, a: *const Mat);
        pub fn mat_eigvals_sym(out: *mut Mat, a: *const Mat);
        pub fn mat_eigen_sym(v: *mut Mat, eigenvalues: *mut Mat, a: *const Mat);
        pub fn mat_eigen(v: *mut Mat, eigenvalues: *mut Mat, a: *const Mat);
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
        Mat(unsafe { ffi::mat_reye(dim) })
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

    /// Returns the number of rows.
    pub fn rows(&self) -> usize {
        unsafe { (*self.0).rows }
    }

    /// Returns the number of columns.
    pub fn cols(&self) -> usize {
        unsafe { (*self.0).cols }
    }

    /// Computes eigenvalues of a general square matrix.
    ///
    /// Uses Hessenberg reduction followed by implicit QR iteration.
    /// For complex eigenvalues (conjugate pairs), only the real part is stored.
    pub fn eigvals(&self) -> Mat {
        let dim = self.rows();
        let out = Mat(unsafe { ffi::mat_vec(dim) });
        unsafe { ffi::mat_eigvals(out.0, self.0) };
        out
    }

    /// Computes eigenvalues of a symmetric matrix.
    ///
    /// Faster than [`eigvals`](Self::eigvals) for symmetric input.
    /// Uses tridiagonal reduction + implicit QR iteration.
    pub fn eigvals_sym(&self) -> Mat {
        let dim = self.rows();
        let out = Mat(unsafe { ffi::mat_vec(dim) });
        unsafe { ffi::mat_eigvals_sym(out.0, self.0) };
        out
    }

    /// Computes eigendecomposition of a symmetric matrix: `A = V * diag(eigenvalues) * V^T`.
    ///
    /// Returns an [`Eigen`] where `eigenvectors` columns are orthogonal eigenvectors
    /// and `eigenvalues` are sorted in ascending order.
    pub fn eigen_sym(&self) -> Eigen {
        let dim = self.rows();
        let v = Mat(unsafe { ffi::mat_mat(dim, dim) });
        let eigenvalues = Mat(unsafe { ffi::mat_vec(dim) });
        unsafe { ffi::mat_eigen_sym(v.0, eigenvalues.0, self.0) };
        Eigen {
            eigenvalues,
            eigenvectors: v,
        }
    }

    /// Computes eigendecomposition of a general square matrix.
    ///
    /// Returns an [`Eigen`] where `eigenvectors` columns are the eigenvectors
    /// and `eigenvalues` holds eigenvalues (real parts for complex conjugate pairs).
    pub fn eigen(&self) -> Eigen {
        let dim = self.rows();
        let v = Mat(unsafe { ffi::mat_mat(dim, dim) });
        let eigenvalues = Mat(unsafe { ffi::mat_vec(dim) });
        unsafe { ffi::mat_eigen(v.0, eigenvalues.0, self.0) };
        Eigen {
            eigenvalues,
            eigenvectors: v,
        }
    }
}

/// Result of an eigendecomposition.
///
/// Contains eigenvalues as a column vector and eigenvectors as columns of a matrix.
pub struct Eigen {
    pub eigenvalues: Mat,
    pub eigenvectors: Mat,
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

    #[test]
    fn test_eigen_sym_identity() {
        let m = Mat::eye(3);
        let eig = m.eigen_sym();
        for i in 0..3 {
            assert!((eig.eigenvalues.at(i, 0) - 1.0).abs() < 1e-5);
        }
        assert!(eig.eigenvectors.equals(&Mat::eye(3)));
    }

    #[test]
    fn test_eigen_sym_diagonal() {
        let m = Mat::from_slice(2, 2, &[3.0, 0.0, 0.0, 7.0]);
        let eig = m.eigen_sym();
        assert!((eig.eigenvalues.at(0, 0) - 3.0).abs() < 1e-5);
        assert!((eig.eigenvalues.at(1, 0) - 7.0).abs() < 1e-5);
    }

    #[test]
    fn test_eigvals_sym() {
        let m = Mat::from_slice(2, 2, &[2.0, 1.0, 1.0, 2.0]);
        let vals = m.eigvals_sym();
        let mut ev = [vals.at(0, 0), vals.at(1, 0)];
        ev.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((ev[0] - 1.0).abs() < 1e-5);
        assert!((ev[1] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_eigvals_general() {
        let m = Mat::eye(2);
        let vals = m.eigvals();
        assert!((vals.at(0, 0) - 1.0).abs() < 1e-5);
        assert!((vals.at(1, 0) - 1.0).abs() < 1e-5);
    }
}
