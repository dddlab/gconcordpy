#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>

using namespace Eigen;

void ccista(

    MatrixXd& S,
    SparseMatrix<double>& X,
    MatrixXd& LambdaMat,
    double lambda2,
    double epstol,
    int    maxitr,
    int    steptype
);


void ccfista(MatrixXd& S,
             SparseMatrix<double>& X,
             MatrixXd& LambdaMat,
             double lambda2,
             double epstol,
             int maxitr,
             int steptype
);


void ccorig(MatrixXd& S,
            MatrixXd& X,
            MatrixXd& LambdaMat,
            double lambda2,
            double epstol,
            int maxitr
);