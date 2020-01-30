#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>

using namespace Eigen;
using namespace std;


double sgn(double val) {
  return (double(0) < val) - (val < double(0));
}

double sthresh(double x, double t ){
  return sgn(x) * max(abs(x)-t, 0.0);
}

void sthreshmat(MatrixXd & x,
                double tau,
                const MatrixXd & t){

  MatrixXd tmp1(x.cols(), x.cols());
  MatrixXd tmp2(x.cols(), x.cols());

  tmp1 = x.array().unaryExpr(ptr_fun(sgn));
  tmp2 = (x.cwiseAbs() - tau*t).cwiseMax(0.0);

  x = tmp1.cwiseProduct(tmp2);

  return;
}


inline double shrink(double a, double b) {
  if (b < fabs(a)) {
    if (a > 0) return(a-b);
    else       return(a+b);
  } else {
    return(0.0);
  }
}

// ista algorithm
void ccista(MatrixXd&       S,
            SparseMatrix<double>& X,
            MatrixXd&       LambdaMat,
            double  lambda2,
            double  epstol,
            int     maxitr,
            int     steptype
              )
{
  int p = S.cols();
  DiagonalMatrix<double, Dynamic> XdiagM(p);
  SparseMatrix<double> Xn;
  SparseMatrix<double> Step;
  MatrixXd W = S * X;
  MatrixXd Wn(p, p);
  MatrixXd G(p, p);
  MatrixXd Gn(p, p);
  MatrixXd subg(p, p);
  MatrixXd tmp(p, p);

  double h = - X.diagonal().array().log().sum() + 0.5 * ((X*W).diagonal().sum()) + (lambda2 * pow(X.norm(), 2));
  double hn = 0;
  double Qn = 0;
  double subgnorm, Xnnorm;
  double tau;
  double taun = 1.0;
  double c = 0.5;
  int itr = 0;
  int loop = 1;
  int diagitr = 0;
  int backitr = 0;

  XdiagM.diagonal() = - X.diagonal();
  G = XdiagM.inverse();
  G += 0.5 * (W + W.transpose());
  if (lambda2 > 0) { G += lambda2 * 2.0 * X; }

  while (loop != 0){

    tau = taun;
    diagitr = 0;
    backitr = 0;

    while ( 1  && (backitr < 100)) { // back-tracking line search

      if (diagitr != 0 || backitr != 0) { tau = tau * c; }

      tmp = MatrixXd(X) - tau*G;
      sthreshmat(tmp, tau, LambdaMat);
      Xn = tmp.sparseView();

      if (Xn.diagonal().minCoeff() < 1e-8 && diagitr < 10) { diagitr += 1; continue; }

      Step = Xn - X;
      Wn = S * Xn;
      Qn = h + Step.cwiseProduct(G).sum() + (1/(2*tau))*pow(Step.norm(),2);
      hn = - Xn.diagonal().cwiseAbs().array().log().sum() + 0.5 * (Xn.cwiseProduct(Wn).sum());

      if (lambda2 > 0) { hn += lambda2 * pow(Xn.norm(), 2); }

      if (hn > Qn) { backitr += 1; } else { break; }

    }

    XdiagM.diagonal() = - Xn.diagonal();
    Gn = XdiagM.inverse();
    Gn += 0.5 * (Wn + Wn.transpose());

    if (lambda2 > 0) { Gn += lambda2 * 2 * MatrixXd(Xn); }

    if ( steptype == 0 ) {
      taun = ( Step * Step ).eval().diagonal().array().sum() / (Step.cwiseProduct( Gn - G ).sum());
    } else if ( steptype == 1 ) {
      taun = 1;
    } else if ( steptype == 2 ){
      taun = tau;
    }

    tmp = MatrixXd(Xn).array().unaryExpr(ptr_fun(sgn));
    tmp = Gn + tmp.cwiseProduct(LambdaMat);
    subg = Gn;
    sthreshmat(subg, 1.0, LambdaMat);
    subg = (MatrixXd(Xn).array() != 0).select(tmp, subg);
    subgnorm = subg.norm();
    Xnnorm = Xn.norm();
    
    X = Xn;
    h = hn;
    G = Gn;

    itr += 1;

    loop = int((itr < maxitr) && (subgnorm/Xnnorm > epstol));

  }

}



// fista algorithm
void ccfista(MatrixXd& S,
             SparseMatrix<double> &X,
             MatrixXd& LambdaMat,
             double lambda2,
             double epstol,
             int maxitr,
             int steptype
               )
{
  int p = S.cols();
  SparseMatrix<double> Theta(X);
  SparseMatrix<double> Xn(p, p);
  SparseMatrix<double> Step(p, p);
  MatrixXd W = S * X;
  MatrixXd Wn(p, p);
  MatrixXd WTh = S * Theta;
  MatrixXd G(p, p);
  MatrixXd Gn(p, p);
  MatrixXd subg(p, p);
  MatrixXd tmp(p, p);

  double hn = 0;
  double hTh = 0;
  double Qn = 0;
  double subgnorm, Xnnorm;
  double tau;
  double taun = 1.0;
  double alpha = 1.0;
  double alphan;
  double c = 0.9;
  int itr = 0;
  int loop = 1;
  int diagitr = 0;
  int backitr = 0;

  G = 0.5 * (WTh + WTh.transpose()) - MatrixXd((MatrixXd((1/Theta.diagonal().array()))).asDiagonal())
    + MatrixXd(2*lambda2*Theta);

  while (loop != 0){
    tau = taun;
    diagitr = 0;
    backitr = 0;

    while ( 1 ) {
      if (diagitr != 0 || backitr != 0) { tau = tau * c; }
      tmp = MatrixXd(Theta) - tau*G;
      sthreshmat(tmp, tau, LambdaMat);
      Xn = tmp.sparseView();
      if (Xn.diagonal().minCoeff() < 1e-8 && diagitr < 50) {
        diagitr += 1;
        continue;
      }

      Step = Xn - Theta;
      Wn = S * Xn;
      hTh = - Theta.diagonal().cwiseAbs().array().log().sum() + 0.5 * (Theta*WTh).trace()
        + lambda2 * pow(Theta.norm(),2);
      Qn = hTh + Step.cwiseProduct(G).sum() + (1/(2*tau))*pow(Step.norm(),2);
      hn = - Xn.diagonal().cwiseAbs().array().log().sum() + 0.5 * (Xn*Wn).trace() + lambda2 * pow(Xn.norm(),2);

      if (hn > Qn) {
        backitr += 1;
      } else {
        break;
      }
    }

    alphan = (1 + sqrt(1 + 4*pow(alpha,2)))/2;

    Theta = Xn + ((alpha - 1)/alphan) * (Xn - X);

    WTh = S * Theta;
    Gn = 0.5 * (WTh + WTh.transpose());
    Gn = Gn - MatrixXd((MatrixXd((1/Theta.diagonal().array()))).asDiagonal())
      + MatrixXd(2*lambda2*Theta);

    if ( steptype == 0 ) {
      taun = (Step * Step).eval().diagonal().array().sum() / (Step * ( Gn - G )).trace();
      if (taun < 0.0) { taun = tau; }
    } else if ( steptype == 1 ) {
      taun = 1;
    } else if ( steptype == 2 ) {
      taun = tau;
    }

    tmp = MatrixXd(Xn).array().unaryExpr(ptr_fun(sgn));
    tmp = Gn + tmp.cwiseProduct(LambdaMat);
    subg = Gn;
    sthreshmat(subg, 1.0, LambdaMat);
    subg = (MatrixXd(Xn).array() != 0).select(tmp, subg);

    subgnorm = subg.norm();
    Xnnorm = Xn.norm();

    alpha = alphan;
    X = Xn; G = Gn;
    itr += 1;

    loop = int((itr < maxitr) && (subgnorm/Xnnorm > epstol));

  }

}


// coordinate-wise descent algorithm
void ccorig(MatrixXd& S,
            MatrixXd& X,
            MatrixXd& LambdaMat,
            double lambda2,
            double epstol,
            int maxitr
              )
{
  int p = S.cols();
  int itr = 0;
  int converged=0;
  double maxdiff;
  double s1, s2, sSum;
  MatrixXd Xold;
  Xold = X;

  while ( (itr<maxitr) && !converged ) {

    maxdiff = 0;

    for (int i=0; i < (p-1); i++){ // loop for off-diagonal elements
      for (int j=i+1 ;j < p ; j++){
        s1 = 0;
        s2 = 0;
        for (int k=0; k < p; k++){
          s1 += X(i,k)*S(j,k);
          s2 += X(k,j)*S(i,k);
        }
        s1 -= X(i,j)*S(j,j);
        s2 -= X(i,j)*S(i,i);

        sSum = S(i,i) + S(j,j) + 4*lambda2;

        X(i,j) = shrink( -(s1+s2), 2*LambdaMat(i,j) )/sSum;
        X(j,i) = X(i,j);
        maxdiff = fmax(maxdiff, fabs(Xold(i,j)-X(i,j)));
        Xold(i,j) = X(i,j);
      }
    }

    for (int i=0; i < p; i++){ // loop for diagonal elements
      s1 = 0;
      for (int k=0; k < p; k++){
        s1 += X(i,k)*S(i,k);
      }
      s1 -= X(i,i)*S(i,i);
      X(i,i) = ( -s1 + sqrt( (s1*s1) + (4*(S(i,i) + 2*lambda2)) ) ) / ( 2*(S(i,i) + 2*lambda2) );
      maxdiff = fmax(maxdiff, fabs(Xold(i,i)-X(i,i)));
      Xold(i,i) = X(i,i);
    }

    if ( maxdiff < epstol ){ converged = 1; }
    itr++;

  }

}