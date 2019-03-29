// Copyright (C) 2008-2016 Johan Jansson and Niclas Jansson as main authors.

// Licensed under the GNU LGPL Version 2.1.
//
// This program solves the incompressible Navier-Stokes equations using
// least-squares-Galerkin stabilized FEM with a Schur-preconditioned fixed-point
// iteration between the momentunm and pressure equations and a do-nothing adaptive
// method.

// This demo was adapted for rotating vertical turbine problem by Van-Dang Nguyen 2016-2019

#include <dolfin.h>
#include <dolfin/main/MPI.h>
#include <dolfin/config/dolfin_config.h>
#include <dolfin/fem/UFC.h>
#ifdef ENABLE_UFL 
#include "ufc2/NSEMomentum3D.h"
#include "ufc2/NSEContinuity3D.h"
#include "ufc2/NSEDualMomentum3D.h"
#include "ufc2/NSEDualContinuity3D.h"
#include "ufc2/NSEErrRepMomentum3D.h"
#include "ufc2/NSEErrRepContinuity3D.h"
#include "ufc2/Drag3D.h"
#include "ufc2/NSEH1.h"
#include "ufc2/NSEH12.h"
#include "ufc2/NSEH1Momentum3D.h"
#include "ufc2/NSEH1Continuity3D.h"
#include "ufc2/NSEH1MomentumGlobal3D.h"
#include "ufc2/NSEH1ContinuityGlobal3D.h"
#include "ufc2/NSEMomentumResidual3D.h"
#include "ufc2/NSEContinuityResidual3D.h"
#include "ufc2/NSEMomentumResidualGlobal3D.h"
#include "ufc2/NSEContinuityResidualGlobal3D.h"
#include "ufc2/NSEErrEst.h"
#include "ufc2/NSEErrEstGlobal.h"
#else
#include "ufc2/NSEMomentum3D.h"
#include "ufc2/NSEContinuity3D.h"
#endif

#include "ufc2/NSE.h"
#include "ufc2/NodeNormal.h"
#include "dolfin/SpaceTimeFunction.h"
#include "ufc2/SlipBC.h"
#include "ufc2/AdaptiveRefinement.h"

#include <ostream>
#include <iomanip>
#include <cstring>
#include <sstream>
#include <string>
#include <algorithm>
#include <map>
#include <mpi.h>

using namespace dolfin;

real bmarg = 1.0e-5 + DOLFIN_EPS;

real adapt_percent = 5.;

real T = 200.0;

real inner_radius = 20.;
real start_angle = 0; // in degrees

real kk1 = 0.1;
real kk2 = 8.0;
real kk3 = 4.0;
real c1 = 0.1;
real nu = 0.0;

real x00lim=0.35;
real BETA_UMAX=0;


real bk = 1.0;

real t_rotate = 100.;

int no_samples = 200;

real speed_per_second = 0.;

bool is_dual = false;

bool is_const_profile = false;

double zmin = -10.0;
double zmax = 10.0;

std::string chkp_file="primal0";
std::string imesh_file="mesh.bin";

bool is_chkp = false;

const long double PI = 3.141592653589793238L;


void computeX(Function& XX, Form* aM, Mesh& mesh)
{
  // Copy mesh coordinates into X array/function
  int d = mesh.topology().dim();
  UFC ufc(aM->form(), mesh, aM->dofMaps());
  Cell c(mesh, 0);
  uint local_dim = c.numEntities(0);
  uint *idx  = new uint[d * local_dim];
  uint *id  = new uint[d * local_dim];
  real *XX_block = new real[d * local_dim];  
  
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    ufc.update(*cell, mesh.distdata());
    (aM->dofMaps())[0].tabulate_dofs(idx, ufc.cell, cell->index());
    
    uint ii = 0;
    uint jj = 0;    
    for(uint i = 0; i < d; i++) 
    {
      for(VertexIterator v(*cell); !v.end(); ++v, ii++) 
      {
		if (!mesh.distdata().is_ghost(v->index(), 0)) 
		{
			XX_block[jj] = v->x()[i];
			id[jj++] = idx[ii];
		}
      }
    }
    XX.vector().set(XX_block, jj, id);
  }
  XX.vector().apply();
  XX.sync_ghosts();
  delete[] XX_block;
  delete[] idx;
  delete[] id;
}

void Rotate(Function& XX, Form* aM, Mesh& mesh, double theta)
{
  MeshGeometry& geometry = mesh.geometry();
  
  uint d = mesh.topology().dim();
  uint N = mesh.numVertices();
  if(dolfin::MPI::numProcesses() > 1)
    N = mesh.distdata().global_numVertices();
  UFC ufc(aM->form(), mesh, aM->dofMaps());
  Cell c(mesh, 0);
  uint local_dim = c.numEntities(0);
  uint *idx  = new uint[d * local_dim];
  uint *id  = new uint[d * local_dim];
  real *XX_block = new real[d * local_dim];  
  
  // Update the mesh
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    ufc.update(*cell, mesh.distdata());
    (aM->dofMaps())[0].tabulate_dofs(idx, ufc.cell, cell->index());

    XX.vector().get(XX_block, d * local_dim, idx);
    
    std::vector<double> xx, yy, zz;
    uint jj = 0;
    for(VertexIterator v(*cell); !v.end(); ++v)
    {
	  for(unsigned int i = 0; i < d; i++)
	  {
		  if (i==0)
			xx.push_back(XX_block[i * local_dim + jj]);
		  if (i==1)
			yy.push_back(XX_block[i * local_dim + jj]);
	  }
      jj++;
    }

    uint j = 0;
    for(VertexIterator v(*cell); !v.end(); ++v)
    {
      Vertex& vertex = *v;
	  for(unsigned int i = 0; i < d; i++)
	  {
		if (i==0)
			XX_block[i * local_dim + j] = xx[j]*cos(theta) - yy[j]*sin(theta);
		if (i==1)
			XX_block[i * local_dim + j] = xx[j]*sin(theta) + yy[j]*cos(theta);
			
	    geometry.x(vertex.index(), i) = XX_block[i * local_dim + j];
	  }
      j++;
    }
  }

  delete[] XX_block;
  delete[] idx;
  delete[] id;

  MPI_Barrier(dolfin::MPI::DOLFIN_COMM);
}

void ComputeTangentialVectors(Mesh& mesh,  Vector& tau_1, 
			      Vector& tau_2, Vector& normal,
			      Form& form, NodeNormal& node_normal)
{
  UFC ufc(form.form(), mesh, form.dofMaps());
  Cell c(mesh, 0);
  uint local_dim = c.numEntities(0);
  uint *idx  = new uint[3 * local_dim];
  uint *id  = new uint[3 * local_dim];
  real *tau_1_block = new real[3 * local_dim];  
  real *tau_2_block = new real[3 * local_dim];  
  real *normal_block = new real[3 * local_dim];

  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    ufc.update(*cell, mesh.distdata());
    
    (form.dofMaps())[1].tabulate_dofs(idx, ufc.cell, cell->index());
    
    uint ii = 0;
    uint jj = 0;    
    for(uint i = 0; i < 3; i++) 
    {
      for(VertexIterator v(*cell); !v.end(); ++v, ii++) 
      {
	if (!mesh.distdata().is_ghost(v->index(), 0)) 
	{
	  tau_1_block[jj] = node_normal.tau_1[i].get(*v);
	  tau_2_block[jj] = node_normal.tau_2[i].get(*v);
	  normal_block[jj] = node_normal.normal[i].get(*v);
	  id[jj++] = idx[ii];
	}
      }
    }

    tau_1.set(tau_1_block, jj, id);
    tau_2.set(tau_2_block, jj, id);
    normal.set(normal_block, jj, id);
  }

  tau_1.apply();
  tau_2.apply();
  normal.apply();
  delete[] tau_1_block;
  delete[] tau_2_block;
  delete[] normal_block;
  delete[] idx;
  delete[] id;

}

// Comparison operator for index/value pairs
struct less_pair : public std::binary_function<std::pair<int, real>,
					       std::pair<int, real>, bool>
{
  bool operator()(std::pair<int, real> x, std::pair<int, real> y)
  {
    return x.second < y.second;
  }
};


void merge(real *a,real *b,real *res,int an,int bn)
{
  real *ap,*bp,*rp;
  ap=a;
  bp=b;
  rp=res;

  while(ap<a+an && bp<b+bn){ 
    if(*ap <= *bp){
      *rp=*ap;
      ap++;
      rp++;
    }
    else { 
      *rp=*bp;
      rp++;
      bp++;
    }
  }
  if(ap<a+an){
    do
      *rp=*ap;
    while(++rp && ++ap<a+an);
  }
  else{
    do
      *rp=*bp;
    while(++rp && ++bp<b+bn);
  }
}

void ComputeLargestIndicators_cell(Mesh& mesh, Vector& e_indx, std::vector<int>& cells,
						  real percentage)
{
  int N = mesh.numCells();
  int M = std::min((int)(N), 
		   (int)((real) 
			 (dolfin::MPI::numProcesses() > 1 ? 
			  mesh.distdata().global_numCells() : mesh.numCells()) * percentage * 0.01));
  
  if(dolfin::MPI::processNumber() == 1)
    dolfin_set("output destination","terminal");
  message("Computing largest indicators");
  message("percentage: %f", percentage);
  message("N: %d", N);
  message("M: %d", M);
  dolfin_set("output destination","silent");


  std::vector<std::pair<int, real> > indicators(N);
  real eind;
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    int id = (*cell).index();
    std::pair<int, real> p;
    p.first = id;
    uint ci = id;    
    if(dolfin::MPI::numProcesses() > 1)
      ci = mesh.distdata().get_cell_global(ci);
    e_indx.get(&eind, 1, &ci);      
    p.second = eind;    
    indicators[id] = p;
  }

  less_pair comp;
  std::sort(indicators.begin(), indicators.end(), comp);


  real *local_eind = new real[M];
  for(int i = 0; i < M; i++)
  {
    std::pair<int, real> p = indicators[N - 1 - i];
    local_eind[M - 1 - i] = p.second;
  }


  /*
   *  FIXME reduce memory usage
   *  merge only half of the recived data
   */

  uint M_max, M_tot;
  MPI_Allreduce(&M, &M_max, 1, MPI_UNSIGNED, MPI_MAX, dolfin::MPI::DOLFIN_COMM);
  MPI_Allreduce(&M, &M_tot, 1, MPI_UNSIGNED, MPI_SUM, dolfin::MPI::DOLFIN_COMM);

  double *recv_eind = new double[M_max];
  double *global_eind = new double[M_tot];
  double *work = new double[M_tot];

  //  std::vector<double> global_eind;

  MPI_Status status;
  uint src,dest;
  uint rank =  dolfin::MPI::processNumber();
  uint size =  dolfin::MPI::numProcesses();
  uint nm = M;
  int num_recv;
  //  global_eind.insert(global_eind.begin(), local_eind, local_eind + M);
  std::memcpy(global_eind, local_eind, M*sizeof(real));

  for(uint i = 1; i < size; i++) {
    src =(rank - i + size) % size;
    dest = (rank + i) % size;

    MPI_Sendrecv(local_eind, M, MPI_DOUBLE, dest, 0, 
		 recv_eind, M_max, MPI_DOUBLE, src, 0, dolfin::MPI::DOLFIN_COMM, &status);
    MPI_Get_count(&status, MPI_DOUBLE,&num_recv);
    //global_eind.insert(global_eind.end(), recv_eind, recv_eind + num_recv);
    merge(recv_eind, global_eind, work, num_recv, nm);
    std::memcpy(global_eind, work, M_tot * sizeof(real));
    nm += num_recv;
    
  }

  //  std::sort(global_eind.begin(), global_eind.end());
  cells.clear();
  int MM = (int)((real) (dolfin::MPI::numProcesses() > 1 ? 
			 mesh.distdata().global_numCells() : mesh.numCells()) * percentage * 0.01);
  int i = 0;
  for(int j = 0; j < MM; j++) {
    if( local_eind[M - 1 - i] >= global_eind[M_tot - 1 - j] ) {
      std::pair<int, real> p = indicators[N - 1 - i];
      cells.push_back(p.first);
      if( (i++) >= std::min(N, MM)) break;    
    }
  }

  dolfin_set("output destination", "terminal");
  message("%d marked cells on cpu %d", cells.size(), dolfin::MPI::processNumber());
  dolfin_set("output destination", "silent");

  
  delete[] local_eind;
  delete[] recv_eind;
  delete[] global_eind;
  delete[] work;
}


void ComputeLargestIndicators_eind(Mesh& mesh, Vector& e_indx, std::vector<int>& cells,
						  real percentage)
{
  int N = mesh.numCells();
  real eind, sum_e, sum_e_local, max_e, max_e_local, min_e, min_e_local;
  sum_e = sum_e_local = max_e_local = 0.0;
  min_e_local = 1e6;
  
  std::vector<std::pair<int, real> > indicators(N);

  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    int id = (*cell).index();
    std::pair<int, real> p;
    p.first = id;
    uint ci = id;    
    if(dolfin::MPI::numProcesses() > 1)
      ci = mesh.distdata().get_cell_global(ci);
    e_indx.get(&eind, 1, &ci);      
    // Take absolute value
    eind = abs(eind);
    p.second = eind;    
    indicators[id] = p;
    max_e_local = std::max(max_e_local, eind);
    min_e_local = std::min(min_e_local, eind);
    sum_e_local += p.second;
  }

  less_pair comp;
  std::sort(indicators.begin(), indicators.end(), comp);

  MPI_Allreduce(&sum_e_local, &sum_e, 1, MPI_DOUBLE,
		MPI_SUM, dolfin::MPI::DOLFIN_COMM);

  MPI_Allreduce(&max_e_local, &max_e, 1, MPI_DOUBLE, 
		MPI_MAX, dolfin::MPI::DOLFIN_COMM);

  MPI_Allreduce(&min_e_local, &min_e, 1, MPI_DOUBLE, 
		MPI_MIN, dolfin::MPI::DOLFIN_COMM);

  real threshold = (percentage * 0.01 * sum_e);
  real cutoff = (max_e + min_e) / 2.0;
  real acc_local, acc;
  acc_local = acc = 0.0;

  int iter = 0;
  while ( (fabs(acc - threshold) / threshold )  > 1e-2  && (iter++) < 10)
  {
    cutoff = (max_e + min_e) / 2.0;
    acc = acc_local = 0.0;
    cells.clear();

    for (int i = 0; i < N; i++) 
    {
      std::pair<int, real> p = indicators[N - 1 - i];

      cells.push_back(p.first);
      acc_local += p.second;

      if ( p.second < cutoff )
	break;     
    }

    MPI_Allreduce(&acc_local, &acc, 1, MPI_DOUBLE, 
		  MPI_SUM, dolfin::MPI::DOLFIN_COMM);
        
    ( acc > threshold ? (min_e = cutoff ) : (max_e = cutoff));    
  }
}

void ComputeRefinementMarkers(Mesh& mesh, real percentage, Vector& e_indx,
			      MeshFunction<bool>& cell_refinement_marker)
{

  real error = 0.0;
  //ComputeError(error);

  //message("err: %g", error);
  
  std::vector<int> cells;
  ComputeLargestIndicators_cell(mesh, e_indx, cells, percentage);
    
  cell_refinement_marker.init(mesh, mesh.topology().dim());
  cell_refinement_marker = false;
    
  int M = cells.size();
       
  for(int i = 0; i < M; i++)
  {
    cell_refinement_marker.set(cells[i], true);
  }

}
//-----------------------------------------------------------------------------
double angle_for_marker = 0.;
double tot_angle = 0.;

int main(int argc, char* argv[])
{

  dolfin_set("output destination","silent");
  if(dolfin::MPI::processNumber() == 0)
    dolfin_set("output destination","terminal");



  Checkpoint chkp;
  std::vector<Function *>func;
  std::vector<Vector *>vec;
  double lastCHKPSaveTime;
  double firstCHKPSaveTime;

  Vector variables_savex;   // to save some variables within checkpoint
                            // 0 -> timestep
                            // lastsample
                            // step counter
                            // total angle
                            // angle for marker
                            // stab counter


  printf("Command: %s", argv[0]);
  
  for (int optind = 1; optind < argc; optind++)
    {
      printf(" %s ",argv[optind]);
      if (argv[optind][0] == '-')
	{
	  switch (argv[optind][1])
	    {
            case 'r':
              inner_radius = atof(argv[optind+1]);
              break;
            case 'a':
              start_angle = atof(argv[optind+1]);
              break;
            case 'k':
              kk1 = atof(argv[optind+1]);
              kk2 = atof(argv[optind+2]);
              kk3 = atof(argv[optind+3]);
              break;
            case 'T':
              T = atof(argv[optind+1]);
              break;
            case 'n':
              no_samples = atoi(argv[optind+1]);
              break;
            case 't':
              speed_per_second = atof(argv[optind+1]);
              break;
            case 'c':
              c1 = atof(argv[optind+1]);
              break;
            case 'd':
              is_dual = atoi(argv[optind+1]);
              break;
            case 'z':
              zmin = atof(argv[optind+1]);
              zmax = atof(argv[optind+2]);
              break;
            case 'g':
              bmarg = atof(argv[optind+1]);
              break;
            case 'v':
              nu = atof(argv[optind+1]);
              break;
            case 'w':
              t_rotate = atof(argv[optind+1]);
              break;
	    case 'x':
	      x00lim = atof(argv[optind+1]);
	      break;
            case 'b':
              bk = atof(argv[optind+1]);
              break;
            case 'p':
              is_const_profile = atoi(argv[optind+1]);
	      if (is_const_profile)
		message("Constant profile");
              break;
	    case 'e':
	      is_chkp = true;
	      chkp_file = argv[optind+1];
	      break;
	    case 'm':
	      imesh_file = argv[optind+1];
	      break;	      
	    }
	}
    }
  printf("\n");

  real primal_T = T;
  real dual_T = 3. * T / 4;

  angle_for_marker = -start_angle*PI/180.;

  // Function for no-slip boundary condition for velocity
  class Noslip : public Function
  {
  public:

    Noslip(Mesh& mesh) : Function(mesh) {}

    void eval(real* values, const real* x) const
    {
      values[0] = 0.0;
      values[1] = 0.0;
      values[2] = 0.0;

      real dd = 16.5;
      real z0 = 0.025;
      real UMAX=log((dd+z0)/ z0);
      real val = 1.;
      if (x[2]<dd+zmin)
	val=log((x[2]-zmin+z0)/ z0)/UMAX;
      values[0] = val;

      if (is_const_profile)
        values[0] = 1.0;

    }
  };

  // Function for no-slip boundary condition for velocity
  class DualNoslip : public Function
  {
  public:

    DualNoslip(Mesh& mesh) : Function(mesh) {}

    void eval(real* values, const real* x) const
    {
      values[0] = 0.0;
      values[1] = 0.0;
      values[2] = 0.0;

      bool cd1 = sqrt(x[0]*x[0]+x[1]*x[1])<inner_radius-bmarg;
      bool cd2 = (x[2]>=zmin + bmarg && x[2]<=zmax - bmarg);

      if (cd1 && cd2)
	values[0] = 1.0;
    }
  };

  // Function for no-slip boundary condition for velocity
  class Outflow : public Function
  {
  public:

    Outflow(Mesh& mesh) : Function(mesh) {}

    void eval(real* values, const real* x) const
    {
      values[0] = 0.0;
    }

  };

  // Sub domain for Dirichlet boundary condition
  class AllBoundary : public SubDomain
  {
    bool inside(const real* x, bool on_boundary) const
    {
      return on_boundary;
    }
  };

  // Sub domain for Dirichlet boundary condition                                                                                                         
  class SlipBoundary : public SubDomain
  {
    bool inside(const real* x, bool on_boundary) const
    {
      bool cd1 = sqrt(x[0]*x[0]+x[1]*x[1])<inner_radius-bmarg;
      bool cd2 = (x[2] >= zmin + bmarg);
      return (cd1 && cd2) && on_boundary;
    }
  };


  // Function for no-slip boundary condition for velocity
  class SlipMarker : public Function
  {
  public:

    SlipMarker(Mesh& mesh) : Function(mesh) {}

    void eval(real* values, const real* x) const
    {
      values[0] = 0.0;
    }
  };

  class InflowBoundary : public SubDomain
  {
    bool inside(const real* x, bool on_boundary) const
    {
      bool cd0 = (x[0] <= 0.0);
      bool cd1 = sqrt(x[0]*x[0]+x[1]*x[1])<inner_radius-bmarg;
      bool cd2 = ( x[2] >= zmin + bmarg );
      return ( cd0 && !cd1 || !cd2) && on_boundary;
    }
  };

  // Sub domain for Dirichlet boundary condition                                                                                                     \
                                                                                                                                                      
  class OutflowBoundary : public SubDomain
  {
    bool inside(const real* x, bool on_boundary) const
    {
      bool cd0 = (x[0] >= 0);
      bool cd1 = sqrt(x[0]*x[0]+x[1]*x[1])<inner_radius-bmarg;
      bool cd3 = x[2]>=zmin + bmarg;
      return ( cd0 && !cd1 && cd3 ) && on_boundary;
    }
  };


  // Function for no-slip boundary condition for velocity
  class PsiMomentum : public Function
  {
  public:

    PsiMomentum(Mesh& mesh) : Function(mesh) {}

    void eval(real* values, const real* x) const
    {
      values[0] = 0.0;
      values[1] = 0.0;
      values[2] = 0.0;
    }

    uint rank() const
    {
      return 1;
    }

    uint dim(uint i) const
    {
      return 3;
    }
  };

  // Function for no-slip boundary condition for velocity
  class PsiContinuity : public Function
  {
  public:

    PsiContinuity(Mesh& mesh) : Function(mesh) {}

    void eval(real* values, const real* x) const
    {
      values[0] = 0.0;
    }

    uint rank() const
    {
      return 0;
    }

    uint dim(uint i) const
    {
      return 0;
    }
  };

  // Function for no-slip boundary condition for velocity
  class BPsiMomentum : public Function
  {
  public:

    BPsiMomentum(Mesh& mesh) : Function(mesh) {}

    void eval(real* values, const real* x) const
    {
      values[0] = 0.0;
      values[1] = 0.0;
      values[2] = 0.0;
    }

    uint rank() const
    {
      return 1;
    }

    uint dim(uint i) const
    {
      return 3;
    }
  };


  // Function for no-slip boundary condition for velocity
  class DTheta : public Function
  {
  public:

    DTheta(Mesh& mesh) : Function(mesh) {}

    void eval(real* values, const real* x) const
    {
      values[0] = 0.0;
      values[1] = 0.0;
      values[2] = 0.0;

      double x00 = x[0]*cos(angle_for_marker) - x[1]*sin(angle_for_marker);
      bool d1 = sqrt(x[0]*x[0]+x[1]*x[1]) < inner_radius - bmarg;
      bool d2 = x00>x00lim;
      bool cd3 = (x[2]>=zmin + bmarg && x[2]<=zmax - bmarg);

      if (d1 && d2 && cd3)
	{
	  values[0] = -1.0;
	}
    }

    uint rank() const
    {
      return 1;
    }

    uint dim(uint i) const
    {
      return 3;
    }
  };

  class LiftMarker : public Function
  {
  public:

    LiftMarker(Mesh& mesh) : Function(mesh) {}

    void eval(real* values, const real* x) const
    {
      values[0] = 0.0;
      values[1] = 0.0;
      values[2] = 0.0;

      double x00 = x[0]*cos(angle_for_marker) - x[1]*sin(angle_for_marker);
      bool d1 = sqrt(x[0]*x[0]+x[1]*x[1])<inner_radius-bmarg;
      bool d2 = x00>x00lim;
      bool cd3 = (x[2]>=zmin + bmarg && x[2]<=zmax - bmarg);                                                                                                     

      if (d1 && d2 && cd3)
	{
	  values[2] = 1.0;
	}
    }

    uint rank() const
    {
      return 1;
    }

    uint dim(uint i) const
    {
      return 3;
    }
  };


  class DragMarker : public Function
  {
  public:

    DragMarker(Mesh& mesh) : Function(mesh) {}

    void eval(real* values, const real* x) const
    {
      values[0] = 0.0;
      values[1] = 0.0;
      values[2] = 0.0;

      double x00 = x[0]*cos(angle_for_marker) - x[1]*sin(angle_for_marker);
      bool d1 = sqrt(x[0]*x[0]+x[1]*x[1])<inner_radius-bmarg;
      bool d2 = x00>x00lim;
      bool cd3 = (x[2]>=zmin + bmarg && x[2]<=zmax - bmarg);                                                                                                     

      if (d1 && d2 && cd3)
	{
	  values[1] = 1.0;
	}
    }

    uint rank() const
    {
      return 1;
    }

    uint dim(uint i) const
    {
      return 3;
    }
  };


  class NormalForceVec : public Function
  {
  public:

    NormalForceVec(Mesh& mesh) : Function(mesh) {}

    void eval(real* values, const real* x) const
    {
      values[0] = 0.0;
      values[1] = 0.0;
      values[2] = 0.0;

      double x00 = x[0]*cos(angle_for_marker) - x[1]*sin(angle_for_marker);
      bool d1 = sqrt(x[0]*x[0]+x[1]*x[1])<inner_radius-bmarg;
      bool d2 = x00>x00lim;
      bool cd3 = (x[2]>=zmin + bmarg && x[2]<=zmax - bmarg);                                                      \


      if (d1 && d2 && cd3)
        {
          values[0] = cos(tot_angle);
          values[1] = sin(tot_angle);
        }
    }

    uint rank() const
    {
      return 1;
    }

    uint dim(uint i) const
    {
      return 3;
    }
  };


  class TangForceVec : public Function
  {
  public:

    TangForceVec(Mesh& mesh) : Function(mesh) {}

    void eval(real* values, const real* x) const
    {
      values[0] = 0.0;
      values[1] = 0.0;
      values[2] = 0.0;

      double x00 = x[0]*cos(angle_for_marker) - x[1]*sin(angle_for_marker);
      bool d1 = sqrt(x[0]*x[0]+x[1]*x[1])<inner_radius-bmarg;
      bool d2 = x00>x00lim;
      bool cd3 = (x[2]>=zmin + bmarg && x[2]<=zmax - bmarg);                                                      \


      if (d1 && d2 && cd3)
        {
          values[0] = -sin(tot_angle);
          values[1] = cos(tot_angle);
        }
    }

    uint rank() const
    {
      return 1;
    }

    uint dim(uint i) const
    {
      return 3;
    }
  };


  class OneVec : public Function
  {
  public:

    OneVec(Mesh& mesh) : Function(mesh) {}

    void eval(real* values, const real* x) const
    {
      values[0] = 1.0;
      values[1] = 1.0;
      values[2] = 1.0;
    }

    uint rank() const
    {
      return 1;
    }

    uint dim(uint i) const
    {
      return 3;
    }
  };

  Mesh mesh(imesh_file);

  if (is_chkp)
    {
      message("Chkp: loading %s",chkp_file.c_str());
      chkp.restart(chkp_file);  
      message("Chkp: loading mesh ...");
      chkp.load(mesh);
      message("Chkp: loading mesh done");
    }

  NSEBilinearForm aa;
  Function beta, beta_u, newpos, oldpos;

  beta.init(mesh, aa, 0);
  beta.vector()=0.0;

  beta_u.init(mesh, aa, 0);
  beta_u.vector()=0.0;


  newpos.init(mesh, aa, 0);
  oldpos.init(mesh, aa, 0);
  computeX(oldpos, &aa, mesh);
  newpos.vector() = 0;

  if (fabs(start_angle)>0)
    {
      //                                                                                                                                              
      Rotate(oldpos, &aa, mesh, start_angle*PI/180.);
      computeX(oldpos, &aa, mesh);
      
      //                                                                                                                                           
    }

  message("Save the mesh to mesh_out.bin");
  File meshfile("mesh_out.bin");
  meshfile << mesh;

  Assembler assembler(mesh);

  message("Running on %d %s", dolfin::MPI::numProcesses(), 
	  (dolfin::MPI::numProcesses() > 1 ? "nodes" : "node"));
  message("Global number of vertices: %d", 
	  (dolfin::MPI::numProcesses() > 1 ? mesh.distdata().global_numVertices() : mesh.numVertices()));
  message("Global number of cells: %d", 
	  (dolfin::MPI::numProcesses() > 1 ? mesh.distdata().global_numCells() : mesh.numCells()));

  message("Global number of vertices: %d", 
	  (dolfin::MPI::numProcesses() > 1 ? mesh.distdata().global_numVertices() : mesh.numVertices()));
  message("Global number of cells: %d", 
	  (dolfin::MPI::numProcesses() > 1 ? mesh.distdata().global_numCells() : mesh.numCells()));

  MeshSize h(mesh);
  // FacetNormal n(mesh);
  NodeNormal nn(mesh);
  CellVolume cv(mesh);

  // Create boundary condition
  Noslip noslip(mesh);
  DualNoslip dnoslip(mesh);
  Outflow outflow(mesh);

  SlipMarker sm(mesh);
  //DirichletBoundary dboundary;
  OutflowBoundary oboundary;
  InflowBoundary iboundary;
  SlipBoundary sboundary;
  AllBoundary aboundary;
  // DirichletBC bc_m(noslip, mesh, dboundary);
  DirichletBC bc_in_m(noslip, mesh, iboundary);
  DirichletBC bc_c(outflow, mesh, oboundary);
  SlipBC slipbc_m(mesh, sboundary, nn);
  DirichletBC bc_dm(dnoslip, mesh, aboundary);

  DirichletBC &bc_in_m0 = bc_in_m;
  DirichletBC &bc_c0 = bc_c;
  SlipBC &slipbc_m0 =  slipbc_m;
  DirichletBC &bc_dm0 = bc_dm;

  //SlipBC bc_dm(mesh, aboundary, nn);

  Array<BoundaryCondition*> bcs_m;
  bcs_m.push_back(&bc_in_m);
  bcs_m.push_back(&slipbc_m);

  Array<BoundaryCondition*> bcs_dm;
  bcs_dm.push_back(&bc_dm);


  uint *c_indices = 0;
  uint *indices = 0;

  real hmin = h.min();
  message("hmin: %f", hmin);
  
  PsiMomentum psim(mesh);
  PsiContinuity psic(mesh);
  BPsiMomentum bpsim(mesh);

  real k = kk1*hmin;
  real c2 = 0.0;
  real c3 = 0.0;

  real rdtol = 5e-2;
  int maxit = 10;

  cout << "c1: " << c1 << endl;
  cout << "c2: " << c2 << endl;

  cout << "nu: " << nu << endl;


  Function u;
  Function u0;
  Function p;
  Function p0;
  Function nuf(mesh, nu);
  Function kf(mesh, k);
  Function k0f(mesh, 0.0);
  Function c1f(mesh, c1);
  Function c2f(mesh, c2);
  Function hminf(mesh, c3*hmin);
  Function umean;
  Function dtu;

  Function rd_u;
  Function rd_p;

  Function up;
  Function pp;
  Function up0;
  Function dtup;

  Function Rm, Rmtot;
  Function Rc, Rctot;
  Function wm, wmtot;
  Function wc, wctot;

  Function ei_m;
  Function ei_c;
  Function eij_m;
  Function eij_c;
  Function eif;

  Function tau_1, tau_2, normal;

  //  Function beta, newpos, oldpos;

  Vector ei;

  // Declare primal and dual forms
  Form *a_m, *L_m, *a_c, *L_c;

  NSEMomentum3DBilinearForm ap_m(u, p, nuf, h, kf, c1f, u0, normal, sm, beta);
  NSEMomentum3DLinearForm Lp_m(u, p, nuf, h, kf, c1f, u0, normal, sm, beta);

  NSEContinuity3DBilinearForm ap_c(h, kf, c1f, hminf);
  NSEContinuity3DLinearForm Lp_c(u, p, h, kf, c1f, u0, p0, hminf, beta);

  //NSEDualMomentum3DBilinearForm ad_m(u, up, nuf, h, kf, c1f, c2f, u0);
  NSEDualMomentum3DBilinearForm ad_m(u, up, p, nuf, h, kf, c1f, c2f, u0,beta);
  NSEDualMomentum3DLinearForm Ld_m(u, up, p, nuf, h, kf, c1f, c2f, u0, psim, bpsim,beta);

  NSEDualContinuity3DBilinearForm ad_c(u, h, kf, c1f, c2f, hminf);
  //NSEDualContinuity3DLinearForm Ld_c(u, up, p, h, kf, c1f, u0, p0, hminf, psic);
  NSEDualContinuity3DLinearForm Ld_c(u, p, h, kf, c1f, c2f, u0, p0, hminf, psic,beta);

  NSEErrRepMomentum3DLinearForm Lrep_m(up, pp, nuf, dtup, u);
  NSEErrRepContinuity3DLinearForm Lrep_c(up, p);

  NSEH1Functional MH1(u, p, h);
  NSEH12Functional MH12(u, p);
  NSEErrEstFunctional MHerrest(h, cv, Rm, Rc, wm, wc);
  NSEErrEstGlobalFunctional MHerrestg(h, cv, Rm, Rc, wm, wc);

  NSEMomentumResidual3DLinearForm LRm(u, p, kf, u0, beta);
  NSEMomentumResidual3DFunctional MRm(h, cv, Rm);
  NSEContinuityResidual3DLinearForm LRc(u, u0);
  NSEContinuityResidual3DFunctional MRc(h, cv, Rc);
  NSEMomentumResidualGlobal3DFunctional MRgm(u, p, h, kf, u0, beta);
  NSEContinuityResidualGlobal3DFunctional MRgc(u, h, u0);

  NSEH1Momentum3DLinearForm Lwm(u, kf, u0);
  NSEH1Momentum3DFunctional Mwm(h, cv, wm);
  NSEH1Continuity3DLinearForm Lwc(p);
  NSEH1Continuity3DFunctional Mwc(h, cv, wc);

  NSEH1MomentumGlobal3DFunctional Mgwm(u, h, kf, u0);
  NSEH1ContinuityGlobal3DFunctional Mgwc(p, h);

  // Initialize functions
  u.init(mesh, ap_m, 0);
  u0.init(mesh, ap_m, 0);
  p.init(mesh, ap_c, 0);
  p0.init(mesh, ap_c, 0);

  rd_u.init(mesh, ap_m, 0);
  rd_p.init(mesh, ap_c, 0);

  up.init(mesh, ap_m, 0);
  up0.init(mesh, ap_m, 0);
  pp.init(mesh, ap_c, 0);
  dtup.init(mesh, ap_m, 0);

  dtu.init(mesh, ap_m, 0);

  Rm.init(mesh, LRm, 0);
  Rc.init(mesh, LRc, 0);
  wm.init(mesh, Lwm, 0);
  wc.init(mesh, Lwc, 0);

  Rmtot.init(mesh, LRm, 0);
  Rctot.init(mesh, LRc, 0);
  wmtot.init(mesh, Lwm, 0);
  wctot.init(mesh, Lwc, 0);

  ei_m.init(mesh, Lrep_m, 0);
  eij_m.init(mesh, Lrep_m, 0);
  ei_c.init(mesh, Lrep_c, 0);
  eij_c.init(mesh, Lrep_c, 0);
  eif.init(mesh, ei, Lrep_c, 0);

  normal.init(mesh, ap_m, 0);
  tau_1.init(mesh, ap_m, 0);
  tau_2.init(mesh, ap_m, 0);

  p.vector() = 1.0;
  p0.vector() = 1.0;

  ei_m.vector() = 0.0;
  ei_c.vector() = 0.0;

  ComputeTangentialVectors(mesh, (Vector&)tau_1.vector(), (Vector&)tau_2.vector(), (Vector&)normal.vector(), ap_m, nn);

  dolfin_set("PDE linear solver", "iterative");

  // Declare PDE solvers

  LinearPDE *pde_m, *pde_c;

  LinearPDE pdep_m(ap_m, Lp_m, mesh, bcs_m);
  LinearPDE pdep_c(ap_c, Lp_c, mesh, bc_c, cg);

  LinearPDE pded_m(ad_m, Ld_m, mesh, bcs_dm);
  LinearPDE pded_c(ad_c, Ld_c, mesh, bc_c);

  Function U;
  Function P;


  // Initialize chkp
  message("Chkp: Initializing ...");
  func.push_back(&u);
  func.push_back(&u0);
  func.push_back(&p);
  func.push_back(&p0);
  func.push_back(&oldpos);
  variables_savex.init(6);
  vec.push_back(&variables_savex);
  double towrite[6];    


  File file_u("velocity.bin");
  File file_p("pressure.bin");
  File file_du("dvelocity.bin");
  File file_dp("dpressure.bin");
  File file_m("mesh.bin");
  File file_normal("normal.bin");

  file_m << mesh;

  int iteration0 = 0;
  int stabcounter = 0;

  real t = 0; real s = 0;


  SpaceTimeFunction* Up = 0;
  SpaceTimeFunction* dtUp = 0;
  SpaceTimeFunction* Pp = 0;
  SpaceTimeFunction* Rmp = 0;
  SpaceTimeFunction* Rcp = 0;

  Up = new SpaceTimeFunction(mesh, up);
  std::vector<std::string> uprimal_fnames;
  Up->util_fileList("velocity_v", no_samples, uprimal_fnames);
  Up->util_addFiles(uprimal_fnames, T);

  dtUp = new SpaceTimeFunction(mesh, dtup);
  std::vector<std::string> dtuprimal_fnames;
  dtUp->util_fileList("dtvelocity_v", no_samples, dtuprimal_fnames);
  dtUp->util_addFiles(dtuprimal_fnames, T);

  Pp = new SpaceTimeFunction(mesh, pp);
  std::vector<std::string> pprimal_fnames;
  Pp->util_fileList("pressure_v", no_samples, pprimal_fnames);
  Pp->util_addFiles(pprimal_fnames, T);

  Rmp = new SpaceTimeFunction(mesh, Rm);
  std::vector<std::string> Rmprimal_fnames;
  Rmp->util_fileList("Rm_v", no_samples, Rmprimal_fnames);
  Rmp->util_addFiles(Rmprimal_fnames, T);

  Rcp = new SpaceTimeFunction(mesh, Rc);
  std::vector<std::string> Rcprimal_fnames;
  Rcp->util_fileList("Rc_v", no_samples, Rcprimal_fnames);
  Rcp->util_addFiles(Rcprimal_fnames, T);

  std::string solver = "primal";

  bool coeffchanged = true;

  real int_errest_gstcs = 0;

  int nsolvers = 1;
  if (is_dual)
    nsolvers = 2;


  for(int solver_idx = 0; solver_idx < nsolvers; solver_idx++)
    {
      if(solver_idx == 1)
	solver = "dual";
    
      if(solver == "dual")
	{
	  c1 = 4.0;
	  c1f.init(mesh, c1);
	}

      if(solver == "primal")
	{
	  cout << "Starting primal solver" << endl;
	  a_m = &ap_m; L_m = &Lp_m; a_c = &ap_c; L_c = &Lp_c;
	  pde_m = &pdep_m; pde_c = &pdep_c;
	}
      else
	{
	  cout << "Starting dual solver" << endl;
	  a_m = &ad_m; L_m = &Ld_m; a_c = &ad_c; L_c = &Ld_c;
	  pde_m = &pded_m; pde_c = &pded_c;
	  T = dual_T;
	}

      u.vector() = 0.0;
      u0.vector() = 0.0;
      p.vector() = 0.0;
      p0.vector() = 0.0;
      up0.vector() = 0.0;

      Rmtot.vector() = 0.0; Rctot.vector() = 0.0; wmtot.vector() = 0.0; wctot.vector() = 0.0;
    
      int stepcounter = 0;
      int sample = 0;
      t = 0;

      real tot_drag = 0;
      int n_mean = 0;

      real tot_H1dualm = 0;
      real tot_H1dualc = 0;
      real tot_H1dualgm = 0;
      real tot_H1dualgc = 0;
      real tot_H1dualgstm = 0;
      real tot_H1dualgstc = 0;
      real tot_H1primal = 0;
      real tot_H1primal2 = 0;

      real tot_Rm = 0;
      real tot_Rc = 0;
      real tot_Rgm = 0;
      real tot_Rgc = 0;
      real tot_Rgstm = 0;
      real tot_Rgstc = 0;

      real int_errest_cs = 0;
      real int_errest_gcs = 0;
    
      k = kk1*hmin;
      kf.init(mesh, k);

      // pde_m->reset();
    
      tot_angle = start_angle*PI/180.; // tot_angle in radians
      message("tot_angle %f",tot_angle);

      ///////////////////////chkp loading///////////////////////////
      if (is_chkp)
	{
	  /*message("Chkp: loading %s",chkp_file.c_str());
	  chkp.restart(chkp_file);  
	  message("Chkp: loading mesh ...");
	  chkp.load(mesh);
	  message("Chkp: loading mesh done");*/

	  lastCHKPSaveTime=0;
	  firstCHKPSaveTime=0;

	  if (chkp.restart())
	    {
	      message("Chkp: loading functions ...");

	      t = chkp.restart_time();
	      lastCHKPSaveTime=t;
	      firstCHKPSaveTime=t;

	      chkp.load(func);
	      chkp.load(vec);
	      message("Chkp: Done loading func and vec.");

	      variables_savex.get(towrite); // load timestep
	      message("Chkp: Done loading variables_savex.");

	      k = towrite[0];
	      sample =  towrite[1];
	      stepcounter =  towrite[2];
	      tot_angle =  towrite[3];
	      angle_for_marker = towrite[4];
	      stabcounter = towrite[5];
	      kf.init(mesh, k);
	    }
	}

      //////////////////////////////////////////////////////////////
      message("Starting the time step at t: %f, k: %f, tot_angle: %f, angle_for_marker: %f",t, k,tot_angle, angle_for_marker);

      // Time-stepping
      while(t <= T)
	{
	  double scale = 1.;
	  if (t<t_rotate)
	    {
	      scale = 0.;
	    }

	  real rotation_angle = scale*speed_per_second*PI/180.*k;

	  if (fabs(rotation_angle)>0)
	    {
	      tot_angle += rotation_angle;
	      angle_for_marker -= rotation_angle;
	      message("Rotating ...");
	      Rotate(oldpos, &ap_m, mesh, rotation_angle);
	      message("ComputeX ...");
	      computeX(newpos, &ap_m, mesh);
	      beta.vector() = newpos.vector();
	      beta.vector() -= oldpos.vector();
	      beta.vector() /= k;
	      beta.vector().apply();
	      beta.sync_ghosts();
	
	      oldpos.vector()=newpos.vector();
	      oldpos.vector().apply();
	      oldpos.sync_ghosts();

	      message("Updating boundary conditions ...");
	      bc_in_m0.update();
	      bc_c0.update();
	      slipbc_m0.update();
	      bc_dm0.update();
	    }

	  real stimer = time();

	  s = primal_T - t;
      
	  if(solver == "dual")
	    {
	      cout << "eval dual" << endl;
	      Up->eval(s);
	      dtUp->eval(s);
	      Pp->eval(s);
	      Rmp->eval(s);
	      Rcp->eval(s);
	      cout << "eval dual done" << endl;
	    }

	  beta_u.vector() = beta.vector();
	  beta_u.vector() -= u.vector();
	  beta_u.vector().apply();
	  beta_u.sync_ghosts();


          real umax = u.vector().norm(linf);
          real beta_umax = beta_u.vector().norm(linf);
          BETA_UMAX = std::max(beta_umax, BETA_UMAX);


	  if(solver == "dual")
	    umax = up.vector().norm(linf);

	  if(stepcounter >= 100)
	    {
	      if(iteration0 > 5)
		stabcounter = 10;
	      k = kk2*hmin/std::max(1.0, BETA_UMAX);

	      if(stabcounter > 0)
		k /= 4.;
	      kf.init(mesh, k);
	    }
	 

	  message("BETA_UMAX: %f, k: %f", BETA_UMAX, k);


	  // Fixed-point iteration
	  for(int i = 0; i < maxit; i++)
	    {
	      message("Solving momentum");
	      real timer = time();
	
	      pde_m->solve(U);
	
       
	      rd_u.vector() = U.vector();
	      rd_u.vector() -= u.vector();
	      real rd_u_norm = rd_u.vector().norm(l2);
	      u.vector() = U.vector();
	
	      cout << "Solving continuity" << endl; 

	      pde_c->solve(P);
	      
	      p.vector() = P.vector();
	      rd_p.vector() = p.vector();
	      rd_p.vector() -= p0.vector();
	      real rd_p_norm = rd_p.vector().norm(l2);
	
	      cout << "Iteration info: " << "Unorm: " << U.vector().norm(linf) << " Pnorm: " << P.vector().norm(linf) << " Uincr: " <<  rd_u_norm << " Pincr: " <<  rd_p_norm << " k: " << k << " step: " << stepcounter << " t: " << t << " timer: " << time() - timer << endl;
	      cout << "iteration: " << i << endl;
	      iteration0 = i;
	      if(rd_u_norm / u.vector().norm(l2) <= rdtol && rd_p_norm / p.vector().norm(l2) <= rdtol)
		{
		  cout << "Step info: " << "Unorm: " << U.vector().norm(linf) << " Pnorm: " << P.vector().norm(linf) << " Uincr: " <<  rd_u_norm / u.vector().norm(l2) << " Pincr: " <<  rd_p_norm / p.vector().norm(l2) << " k: " << k << " step: " << stepcounter << " iters: " << iteration0 + 1 << " t: " << t << " timer: " << time() - stimer << endl;
		  break;
		}
	      p0.vector() = p.vector();
	    }

	    pde_c->reset();

      
	  if(solver == "dual")
	    {
	      cout << "errest" << endl;
	      assembler.assemble(eij_m.vector(), Lrep_m);
	      ei_m.vector().axpy(k, eij_m.vector());
	      assembler.assemble(eij_c.vector(), Lrep_c);
	      ei_c.vector().axpy(k, eij_c.vector());
	      cout << "errest done: " << ei_m.vector().norm(linf) << " " << ei_c.vector().norm(linf) << endl;

	      assembler.assemble(wm.vector(), Lwm);
	      real H1dualm = assembler.assemble(Mwm);
	      tot_H1dualm += k*H1dualm;
	      real H1dualgm = sqrt(assembler.assemble(Mgwm));
	      tot_H1dualgm += k*H1dualgm;
	      real H1dualgstm = assembler.assemble(Mgwm);
	      tot_H1dualgstm += k*H1dualgstm;
	      wmtot.vector().axpy(k, wm.vector());
	      assembler.assemble(wc.vector(), Lwc);
	      real H1dualc = assembler.assemble(Mwc);
	      tot_H1dualc += k*H1dualc;
	      real H1dualgc = sqrt(assembler.assemble(Mgwc));
	      tot_H1dualgc += k*H1dualgc;
	      real H1dualgstc = assembler.assemble(Mgwc);
	      tot_H1dualgstc += k*H1dualgstc;
	      wctot.vector().axpy(k, wc.vector());


	      real Rmi = assembler.assemble(MRm);
	      tot_Rm += k*Rmi;

	      real Rgmi = 0.0;
	      Rmtot.vector().axpy(k, Rm.vector());
	      real Rci = assembler.assemble(MRc);
	      tot_Rc += k*Rci;

	      real Rgci = 0.0;
	      Rctot.vector().axpy(k, Rc.vector());

	      real errest_cs = assembler.assemble(MHerrest);
	      int_errest_cs += k*errest_cs;
	      real errest_gcs = sqrt(assembler.assemble(MHerrestg));
	      int_errest_gcs += k*errest_gcs;
	      real errest_gstcs = assembler.assemble(MHerrestg);
	      int_errest_gstcs += k*errest_gstcs;
	      n_mean++;

	      cout << "step dual t: " << t <<
		" dualm: " << H1dualm <<
		" dualc: " << H1dualc <<
		" dualgm: " << H1dualgm <<
		" dualgc: " << H1dualgc <<
		" Rm: " << Rmi <<
		" Rc: " << Rci <<
		" Rgm: " << Rgmi <<
		" Rgc: " << Rgci <<
		" errest_cs: " << errest_cs <<
		" errest_gcs: " << errest_gcs <<
		endl;
	    }

	  if(solver == "primal")
	    {
	      nn.__compute_normal(mesh);
	      ComputeTangentialVectors(mesh, (Vector&)tau_1.vector(), (Vector&)tau_2.vector(), (Vector&)normal.vector(), ap_m, nn);
	      DTheta dtheta(mesh);

	      NormalForceVec normalforcevec(mesh);
              TangForceVec tangforcevec(mesh);


	      Drag3DFunctional Df(normal, dtheta, p);

	      Drag3DFunctional Nf(normalforcevec, normal, p);
              Drag3DFunctional Tf(tangforcevec, normal, p);
                                
	      Function one(mesh, 1.0);                                                                                                       	
	      Drag3DFunctional SurfaceArea(dtheta, dtheta, one);

	      real drag = assembler.assemble(Df);

	      real surface_area = assembler.assemble(SurfaceArea);

	      real normalforce = assembler.assemble(Nf);
              real tangforce = assembler.assemble(Tf);

              message("normal force: %f, tangential force: %f, angle: %.16f, t = %.16f\n", normalforce, tangforce, tot_angle, t);

	      assembler.assemble(Rm.vector(), LRm);
	      real Rmi = assembler.assemble(MRm);
	      cout << "step primal Rm: " << Rmi << endl;
	      assembler.assemble(Rc.vector(), LRc);
	      real Rci = assembler.assemble(MRc);
	      cout << "step primal Rc: " << Rci << endl;

	      if (t >= dual_T)
		{
		  tot_drag = (drag + n_mean*tot_drag) / (n_mean + 1);
		  //cout << "step mean drag: " << tot_drag << endl;
		  cout << "step t: " << t << " drag: " << drag << endl;
		  real H1primal = assembler.assemble(MH1);
		  tot_H1primal = (H1primal + n_mean*tot_H1primal) / (n_mean + 1);
		  //tot_H1primal += k*H1primal;
		  real H1primal2 = assembler.assemble(MH12);
		  tot_H1primal2 = (H1primal2 + n_mean*tot_H1primal2) / (n_mean + 1);
		  //tot_H1primal2 += k*H1primal2;
		  cout << "step H1 primal: " << tot_H1primal << endl;
		  cout << "step H1 primal2: " << tot_H1primal2 << endl;
		  n_mean++;

		  real Rgstmi = assembler.assemble(MRgm);
		  tot_Rgstm += k*Rgstmi;
		  real Rgstci = assembler.assemble(MRgc);
		  tot_Rgstc += k*Rgstci;
		}
	    }

	  if(stepcounter == 0 || t > T*(real(sample)/real(no_samples)))
	    {
	      if(solver == "primal")
		{
		  file_u << u; file_p << p; file_normal << normal;
	  
		  // Record primal solution
		  std::stringstream number;
		  number << std::setfill('0') << std::setw(6) << sample;
	  
		  // Save primal velocity
		  up.vector() = u.vector(); up.vector() += u0.vector(); up.vector() /= 2.;
		  std::stringstream ufilename;
		  ufilename << "velocity_v" << number.str() << ".bin" << std::ends;
		  File ubinfile(ufilename.str());
		  ubinfile << up.vector();

		  // Save mesh
		  std::stringstream mfilename;
		  mfilename << "mesh" << number.str() << ".bin" << std::ends;
		  File mbinfile(mfilename.str());
		  mbinfile << mesh;

		  // Save primal velocity time-derivative
		  dtu.vector() = u.vector(); dtu.vector() -= u0.vector(); dtu.vector() /= k;
		  std::stringstream dtufilename;
		  dtufilename << "dtvelocity_v" << number.str() << ".bin" << std::ends;
		  File dtubinfile(dtufilename.str());
		  dtubinfile << dtu.vector();
	  
		  // Save primal pressure
		  std::stringstream pfilename;
		  pfilename << "pressure_v" << number.str() << ".bin" << std::ends;
		  File pbinfile(pfilename.str());
		  pbinfile << p.vector();

		  // Save primal residuals
		  std::stringstream Rmfilename;
		  Rmfilename << "Rm_v" << number.str() << ".bin" << std::ends;
		  File Rmbinfile(Rmfilename.str());
		  Rmbinfile << Rm.vector();

		  std::stringstream Rcfilename;
		  Rcfilename << "Rc_v" << number.str() << ".bin" << std::ends;
		  File Rcbinfile(Rcfilename.str());
		  Rcbinfile << Rc.vector();
		}
	      else
		{
		  file_du << u; file_dp << p;
		}
	

	      ///////////////////////chkp saving////////////////////////////                                                                                                                                   
	      message("Chkp: saving ...");
	      towrite[0] = k;
	      towrite[1] = sample + 1;                  
	      towrite[2] = stepcounter + 1; 
	      towrite[3] = tot_angle;
	      towrite[4] = angle_for_marker;  
	      towrite[5] = stabcounter;            
	      variables_savex.set(towrite);
	      chkp.write(solver, solver=="dual", t+k, mesh, func, vec);     
	      lastCHKPSaveTime=t;

	      //////////////////////////////////////////////////////////////                                                                                                

	      sample++;
	    }
      
	  u0.vector() = u.vector();
	  up0.vector() = up.vector();


	  stepcounter++;
	  if(stabcounter > 0)
	    stabcounter--;

	  t += k;
	}
    

      cout << "Solver done" << endl;

      if(solver == "primal")
	{
	  cout << "mean drag: " << tot_drag << endl;
	  cout << "total H1primal: " << sqrt(tot_H1primal) << endl;
	  cout << "total H1primal2: " << sqrt(tot_H1primal2) << endl;
	  cout << "total Rgstm: " << sqrt(tot_Rgstm) << endl;
	  cout << "total Rgstc: " << sqrt(tot_Rgstc) << endl;

	  int_errest_gstcs = (sqrt(tot_Rgstm) + sqrt(tot_Rgstc));
	}

      if(solver == "dual")
	{
	  cout << "Preparing adaptivity" << endl;
	  // Adaptive error control
	  if(!ParameterSystem::parameters.defined("adapt_algorithm"))
	    dolfin_add("adapt_algorithm", "rivara");
	  dolfin_set("adapt_algorithm", "rivara");
	  if(!ParameterSystem::parameters.defined("output_format"))
	    dolfin_add("output_format", "binary");
	  dolfin_set("output_format", "binary");
	  MeshFunction<bool> cell_marker;

      
	  eif.vector() = 0.0;
	  eif.vector() += ei_m.vector(); eif.vector() += ei_c.vector();

      
	  NSEErrRepMomentum3DFunctional M_ei(eif, cv);
	  real errest = fabs(assembler.assemble(M_ei));

	  int_errest_gstcs *= (sqrt(tot_H1dualgstm) + sqrt(tot_H1dualgstc));

	  cout << "error estimate: " << errest / dual_T << endl;
	  cout << "error estimate cs: " << int_errest_cs / dual_T << endl;
	  cout << "error estimate gcs: " << int_errest_gcs / dual_T << endl;
	  cout << "error estimate gstcs: " << int_errest_gstcs / dual_T << endl;
	  cout << "total H1dualm: " << tot_H1dualm << endl;
	  cout << "total H1dualc: " << tot_H1dualc << endl;
	  cout << "total H1dualgm: " << tot_H1dualgm << endl;
	  cout << "total H1dualgc: " << tot_H1dualgc << endl;
	  cout << "total H1dualgstm: " << sqrt(tot_H1dualgstm) << endl;
	  cout << "total H1dualgstc: " << sqrt(tot_H1dualgstc) << endl;
	  cout << "total Rm: " << tot_Rm << endl;
	  cout << "total Rc: " << tot_Rc << endl;
	  cout << "total Rgm: " << tot_Rgm << endl;
	  cout << "total Rgc: " << tot_Rgc << endl;

	  File file_ei("ei.bin");
	  file_ei << eif;

	  File file_Rmtot("Rmtot.bin");
	  file_Rmtot << Rmtot;
	  File file_Rctot("Rctot.bin");
	  file_Rctot << Rctot;
	  File file_wmtot("wmtot.bin");
	  file_wmtot << wmtot;
	  File file_wctot("wctot.bin");
	  file_wctot << wctot;

	  MeshFunction<real> eimf;
	  eimf.init(mesh, mesh.topology().dim());
  
	  // Initialize eimf - assumption on dofmap for DG0
	  for (CellIterator c(mesh); !c.end(); ++c)
	    {
	      eimf.set(*c, eif.vector()[c->index()]);
	    }

	  MPI_Barrier(dolfin::MPI::DOLFIN_COMM);
  
	  cout << "Output eimf: " << endl;
	  File file_eimf("eimf.bin");
	  file_eimf << eimf;
      

	  ComputeRefinementMarkers(mesh, adapt_percent, ei, cell_marker);
	  AdaptiveRefinement::refine(mesh, cell_marker);

	  File file_rm("rmesh.bin");
	  file_rm << mesh;
	}

    }
  
  return 0;
}
