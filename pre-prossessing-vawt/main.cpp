// Copyright (C) 2015-2019 Van-Dang Nguyen

// Licensed under the GNU LGPL Version 2.1.
//
// This program is used to pre-process the VAWT meshes
//
//

#include <dolfin.h>
#include <dolfin/main/MPI.h>
#include <dolfin/config/dolfin_config.h>
#include <dolfin/fem/UFC.h>

#include "ufc2/NSE.h"
#include "ufc2/NodeNormal.h"
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

real robj = 9.0;
real Lx = 9.0;
real Ly = 9.0;
real Lz = 9.0;

real x00 = 0.0;
real y00 = 0.0;
real z00 = 0.0;

int nref = 0;
real kk1 = 0.9;
real kk2 = 0.01;

real start_angle = 0; // in degrees                                                                                                                                                                                                                                                                                                
std::string shape;
std::string imesh_file = "mesh.bin";
std::string omesh_file = "rmesh.bin";

std::string rmethod="adaptive";

const long double PI = 3.141592653589793238L;

void ComputeSimpleRefinementMarkers(Mesh& mesh,MeshFunction<bool>& cell_refinement_marker, double robj, double hmin, double hmax)
{
  cell_refinement_marker.init(mesh, mesh.topology().dim());
  cell_refinement_marker = false;

  double havg = hmin + kk1*(hmax-hmin);

  for (CellIterator cell(mesh); !cell.end(); ++cell)
    {
      int id = (*cell).index();
      real h0 = (*cell).diameter();
      Point p = (*cell).midpoint();
      bool cd1 = sqrt((p.x() - x00)*(p.x() - x00) + (p.y() - y00)*(p.y() - y00)) < robj && fabs(p.z() - z00) < robj;

      if (shape=="sphere")
	cd1 = sqrt((p.x() - x00)*(p.x() - x00) + (p.y() - y00)*(p.y() - y00) + (p.z() - z00)*(p.z() - z00) ) < robj;

      if (shape=="box")
	cd1 = fabs(p.x() - x00)<=Lx && fabs(p.y() - y00)<=Ly && fabs(p.z() - z00)  < Lz;

      if (shape=="zcylinder")
	cd1 = sqrt((p.x() - x00)*(p.x() - x00) + (p.y() - y00)*(p.y() - y00))<robj && fabs(p.z() - z00) < Lz;

      if (shape=="xcylinder")
        cd1 = sqrt((p.z() - z00)*(p.z() - z00) + (p.y() - y00)*(p.y() - y00))<robj && fabs(p.x() - x00) < Lx;

      if (shape=="ycylinder")
        cd1 = sqrt((p.x() - x00)*(p.x() - x00) + (p.z() - z00)*(p.z() - z00))<robj && fabs(p.y() - y00) < Ly;
  
      bool cd2 = h0> hmin + kk2*(hmax-hmin);
      bool cd3 = h0>havg;

      if ( (cd1 && cd2) || cd3 )
	{
	  cell_refinement_marker.set(id, true);
	}
    }
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


int main(int argc, char* argv[])
{
  dolfin_set("output destination","silent");
  if(dolfin::MPI::processNumber() == 0)
    dolfin_set("output destination","terminal");

  message("Command: %s", argv[0]);
  
  for (int optind = 1; optind < argc; optind++)
    {
      message(" %s ",argv[optind]);
      if (argv[optind][0] == '-')
	{
	  switch (argv[optind][1])
	    {
            case 's':
              shape = argv[optind+1];
              break;
            case 'm':
              imesh_file = argv[optind+1];
              break;
            case 'o':
              omesh_file = argv[optind+1];
              break;
	    case 'n':
	      nref = atoi(argv[optind+1]);
	      break;
            case 'r':
              robj = atof(argv[optind+1]);
              break;
            case 'L':
              Lx = atof(argv[optind+1]);
              Ly = atof(argv[optind+2]);
              Lz = atof(argv[optind+3]);
              break;
            case 'c':
              x00 = atof(argv[optind+1]);
              y00 = atof(argv[optind+2]);
              z00 = atof(argv[optind+3]);
              break;
            case 'k':
              kk1 = atof(argv[optind+1]);
              kk2 = atof(argv[optind+2]);
              break;
            case 'a':
              rmethod = argv[optind+1];
              break;
            case 't':
              start_angle = atof(argv[optind+1]);
              break;	    
	    }
	}
    }

   // Create mesh
  Mesh mesh(imesh_file);
  
  message("Global number of vertices: %d",
	  (dolfin::MPI::numProcesses() > 1 ? mesh.distdata().global_numVertices() : mesh.numVertices()));
  message("Global number of cells: %d",
	  (dolfin::MPI::numProcesses() > 1 ? mesh.distdata().global_numCells() : mesh.numCells()));

  message("Refine %d times", nref);


  message("Rotate: %f degree", start_angle);
  NSEBilinearForm aa;
  Function beta, newpos, oldpos;

  beta.init(mesh, aa, 0);
  beta.vector()=0.0;

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


  cout << "Preparing adaptivity" << endl;
  // Adaptive error control                                                                                                                                        
  if(!ParameterSystem::parameters.defined("adapt_algorithm"))
    dolfin_add("adapt_algorithm", "rivara");
  dolfin_set("adapt_algorithm", "rivara");
  if(!ParameterSystem::parameters.defined("output_format"))
    dolfin_add("output_format", "binary");
  dolfin_set("output_format", "binary");

  for (int iref = 0; iref<nref; iref ++)
  {
    if(dolfin::MPI::processNumber() == 0)
      dolfin_set("output destination","terminal");

    MeshSize h(mesh);
    message("hmin: %f, hmax: %f",h.min(), h.max());
    message("Refining mesh ...");
    MeshFunction<bool> cell_marker;
    ComputeSimpleRefinementMarkers(mesh, cell_marker, robj, h.min(), h.max());
    if (rmethod=="adaptive")
    {
      message("Using AdaptiveRefinement");
      AdaptiveRefinement::refine(mesh, cell_marker);
    }
    else
    {
      message("Using RivaraRefinement");
      RivaraRefinement::refine(mesh, cell_marker);
    }
    message("done");
  }

  MeshSize h(mesh);
  message("hmin: %f, hmax: %f",h.min(), h.max());

  File file_rm(omesh_file);
  file_rm << mesh;
  message("vertices after: %d",
	  (dolfin::MPI::numProcesses() > 1 ? mesh.distdata().global_numVertices() : mesh.numVertices()));
  message("cells after: %d",
	  (dolfin::MPI::numProcesses() > 1 ? mesh.distdata().global_numCells() : mesh.numCells()));


  NodeNormal nn(mesh);

  Function tau_1, tau_2, normal;

  NSEBilinearForm a;

  normal.init(mesh, a, 0);
  tau_1.init(mesh, a, 0);
  tau_2.init(mesh, a, 0);

  ComputeTangentialVectors(mesh, (Vector&)tau_1.vector(), (Vector&)tau_2.vector(), (Vector&)normal.vector(), a, nn);

  File file_normal("normal_test.bin");

  file_normal << normal;

  message("Global number of vertices: %d",
          (dolfin::MPI::numProcesses() > 1 ? mesh.distdata().global_numVertices() : mesh.numVertices()));
  message("Global number of cells: %d",
          (dolfin::MPI::numProcesses() > 1 ? mesh.distdata().global_numCells() : mesh.numCells()));

  return 0;
}
