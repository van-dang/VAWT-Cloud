// This code conforms with the UFC specification version 2.0.5
// and was automatically generated by FFC version 1.0.0.
//
// This code was generated with the option '-l dolfin' and
// contains DOLFIN-specific wrappers that depend on DOLFIN.
// 
// This code was generated with the following parameters:
// 
//   cache_dir:                      ''
//   convert_exceptions_to_warnings: False
//   cpp_optimize:                   False
//   cpp_optimize_flags:             '-O2'
//   epsilon:                        1e-14
//   error_control:                  False
//   form_postfix:                   True
//   format:                         'dolfin'
//   log_level:                      20
//   log_prefix:                     ''
//   optimize:                       True
//   output_dir:                     '.'
//   precision:                      15
//   quadrature_degree:              'auto'
//   quadrature_rule:                'auto'
//   representation:                 'quadrature'
//   split:                          True
//   swig_binary:                    'swig'
//   swig_path:                      ''

#ifndef __NSEERRESTGLOBAL_H
#define __NSEERRESTGLOBAL_H

#include <cmath>
#include <stdexcept>
#include <fstream>
#include <ufc.h>

/// This class defines the interface for a finite element.

class nseerrestglobal_finite_element_0: public ufc::finite_element
{
public:

  /// Constructor
  nseerrestglobal_finite_element_0();

  /// Destructor
  virtual ~nseerrestglobal_finite_element_0();

  /// Return a string identifying the finite element
  virtual const char* signature() const;

  /// Return the cell shape
  virtual ufc::shape cell_shape() const;

#ifndef UFC_BACKWARD_COMPATIBILITY
  /// Return the topological dimension of the cell shape
  virtual unsigned int topological_dimension() const;

  /// Return the geometric dimension of the cell shape
  virtual unsigned int geometric_dimension() const;
#endif
  /// Return the dimension of the finite element function space
  virtual unsigned int space_dimension() const;

  /// Return the rank of the value space
  virtual unsigned int value_rank() const;

  /// Return the dimension of the value space for axis i
  virtual unsigned int value_dimension(unsigned int i) const;

  /// Evaluate basis function i at given point in cell
  virtual void evaluate_basis(unsigned int i,
                              double* values,
                              const double* coordinates,
                              const ufc::cell& c) const;

  /// Evaluate all basis functions at given point in cell
  virtual void evaluate_basis_all(double* values,
                                  const double* coordinates,
                                  const ufc::cell& c) const;

  /// Evaluate order n derivatives of basis function i at given point in cell
  virtual void evaluate_basis_derivatives(unsigned int i,
                                          unsigned int n,
                                          double* values,
                                          const double* coordinates,
                                          const ufc::cell& c) const;

  /// Evaluate order n derivatives of all basis functions at given point in cell
  virtual void evaluate_basis_derivatives_all(unsigned int n,
                                              double* values,
                                              const double* coordinates,
                                              const ufc::cell& c) const;

  /// Evaluate linear functional for dof i on the function f
  virtual double evaluate_dof(unsigned int i,
                              const ufc::function& f,
                              const ufc::cell& c) const;

  /// Evaluate linear functionals for all dofs on the function f
  virtual void evaluate_dofs(double* values,
                             const ufc::function& f,
                             const ufc::cell& c) const;

  /// Interpolate vertex values from dof values
  virtual void interpolate_vertex_values(double* vertex_values,
                                         const double* dof_values,
                                         const ufc::cell& c) const;

#ifndef UFC_BACKWARD_COMPATIBILITY
  /// Map coordinate xhat from reference cell to coordinate x in cell
  virtual void map_from_reference_cell(double* x,
                                       const double* xhat,
                                       const ufc::cell& c) const;

  /// Map from coordinate x in cell to coordinate xhat in reference cell
  virtual void map_to_reference_cell(double* xhat,
                                     const double* x,
                                     const ufc::cell& c) const;
#endif

  /// Return the number of sub elements (for a mixed element)
  virtual unsigned int num_sub_elements() const;

  /// Create a new finite element for sub element i (for a mixed element)
  virtual ufc::finite_element* create_sub_element(unsigned int i) const;

#ifndef UFC_BACKWARD_COMPATIBILITY
  /// Create a new class instance
  virtual ufc::finite_element* create() const;

#endif
};

/// This class defines the interface for a local-to-global mapping of
/// degrees of freedom (dofs).

#ifndef UFC_BACKWARD_COMPATIBILITY
class nseerrestglobal_dofmap_0: public ufc::dofmap
#else
class nseerrestglobal_dofmap_0: public ufc::dof_map
#endif
{
private:

  unsigned int _global_dimension;
public:

  /// Constructor
  nseerrestglobal_dofmap_0();

  /// Destructor
  virtual ~nseerrestglobal_dofmap_0();

  /// Return a string identifying the dofmap
  virtual const char* signature() const;

  /// Return true iff mesh entities of topological dimension d are needed
  virtual bool needs_mesh_entities(unsigned int d) const;

  /// Initialize dofmap for mesh (return true iff init_cell() is needed)
  virtual bool init_mesh(const ufc::mesh& m);

  /// Initialize dofmap for given cell
  virtual void init_cell(const ufc::mesh& m,
                         const ufc::cell& c);

  /// Finish initialization of dofmap for cells
  virtual void init_cell_finalize();

#ifndef UFC_BACKWARD_COMPATIBILITY
  /// Return the topological dimension of the associated cell shape
  virtual unsigned int topological_dimension() const;

  /// Return the geometric dimension of the associated cell shape
  virtual unsigned int geometric_dimension() const;
#endif

  /// Return the dimension of the global finite element function space
  virtual unsigned int global_dimension() const;

#ifndef UFC_BACKWARD_COMPATIBILITY
  /// Return the dimension of the local finite element function space for a cell
  virtual unsigned int local_dimension(const ufc::cell& c) const;

  /// Return the maximum dimension of the local finite element function space
  virtual unsigned int max_local_dimension() const;
#else

  /// Return the dimension of the local finite element function space for a cell
  virtual unsigned int local_dimension() const;

  /// Return the maximum dimension of the local finite element function space
  virtual unsigned int geometric_dimension() const;
#endif
  /// Return the number of dofs on each cell facet
  virtual unsigned int num_facet_dofs() const;

  /// Return the number of dofs associated with each cell entity of dimension d
  virtual unsigned int num_entity_dofs(unsigned int d) const;

  /// Tabulate the local-to-global mapping of dofs on a cell
  virtual void tabulate_dofs(unsigned int* dofs,
                             const ufc::mesh& m,
                             const ufc::cell& c) const;

  /// Tabulate the local-to-local mapping from facet dofs to cell dofs
  virtual void tabulate_facet_dofs(unsigned int* dofs,
                                   unsigned int facet) const;

  /// Tabulate the local-to-local mapping of dofs on entity (d, i)
  virtual void tabulate_entity_dofs(unsigned int* dofs,
                                    unsigned int d, unsigned int i) const;

  /// Tabulate the coordinates of all dofs on a cell
  virtual void tabulate_coordinates(double** coordinates,
                                    const ufc::cell& c) const;

#ifndef UFC_BACKWARD_COMPATIBILITY
  /// Return the number of sub dofmaps (for a mixed element)
  virtual unsigned int num_sub_dofmaps() const;

  /// Create a new dofmap for sub dofmap i (for a mixed element)
  virtual ufc::dofmap* create_sub_dofmap(unsigned int i) const;

  /// Create a new class instance
  virtual ufc::dofmap* create() const;
#else
  /// Return the number of sub dofmaps (for a mixed element)
  virtual unsigned int num_sub_dof_maps() const;

  /// Create a new dofmap for sub dofmap i (for a mixed element)
  virtual ufc::dof_map* create_sub_dof_map(unsigned int i) const;
#endif
};

/// This class defines the interface for the tabulation of the cell
/// tensor corresponding to the local contribution to a form from
/// the integral over a cell.

class nseerrestglobal_cell_integral_0_0: public ufc::cell_integral
{
public:

  /// Constructor
  nseerrestglobal_cell_integral_0_0();

  /// Destructor
  virtual ~nseerrestglobal_cell_integral_0_0();

  /// Tabulate the tensor for the contribution from a local cell
  virtual void tabulate_tensor(double* A,
                               const double * const * w,
                               const ufc::cell& c) const;

 #ifndef UFC_BACKWARD_COMPATIBILITY 
  /// Tabulate the tensor for the contribution from a local cell
  /// using the specified reference cell quadrature points/weights
  virtual void tabulate_tensor(double* A,
                               const double * const * w,
                               const ufc::cell& c,
                               unsigned int num_quadrature_points,
                               const double * const * quadrature_points,
                               const double* quadrature_weights) const;
#endif
};

/// This class defines the interface for the assembly of the global
/// tensor corresponding to a form with r + n arguments, that is, a
/// mapping
///
///     a : V1 x V2 x ... Vr x W1 x W2 x ... x Wn -> R
///
/// with arguments v1, v2, ..., vr, w1, w2, ..., wn. The rank r
/// global tensor A is defined by
///
///     A = a(V1, V2, ..., Vr, w1, w2, ..., wn),
///
/// where each argument Vj represents the application to the
/// sequence of basis functions of Vj and w1, w2, ..., wn are given
/// fixed functions (coefficients).

class nseerrestglobal_form_0: public ufc::form
{
public:

  /// Constructor
  nseerrestglobal_form_0();

  /// Destructor
  virtual ~nseerrestglobal_form_0();

  /// Return a string identifying the form
  virtual const char* signature() const;

  /// Return the rank of the global tensor (r)
  virtual unsigned int rank() const;

  /// Return the number of coefficients (n)
  virtual unsigned int num_coefficients() const;

 #ifndef UFC_BACKWARD_COMPATIBILITY 
  /// Return the number of cell domains
  virtual unsigned int num_cell_domains() const;

  /// Return the number of exterior facet domains
  virtual unsigned int num_exterior_facet_domains() const;

  /// Return the number of interior facet domains
  virtual unsigned int num_interior_facet_domains() const;
#else
  /// Return the number of cell domains
  virtual unsigned int num_cell_integrals() const;

  /// Return the number of exterior facet domains
  virtual unsigned int num_exterior_facet_integrals() const;

  /// Return the number of interior facet domains
  virtual unsigned int num_interior_facet_integrals() const;

#endif
  /// Create a new finite element for argument function i
  virtual ufc::finite_element* create_finite_element(unsigned int i) const;

 #ifndef UFC_BACKWARD_COMPATIBILITY 
  /// Create a new dofmap for argument function i
  virtual ufc::dofmap* create_dofmap(unsigned int i) const;
#else
  /// Create a new dofmap for argument function i
  virtual ufc::dof_map* create_dof_map(unsigned int i) const;

#endif
  /// Create a new cell integral on sub domain i
  virtual ufc::cell_integral* create_cell_integral(unsigned int i) const;

  /// Create a new exterior facet integral on sub domain i
  virtual ufc::exterior_facet_integral* create_exterior_facet_integral(unsigned int i) const;

  /// Create a new interior facet integral on sub domain i
  virtual ufc::interior_facet_integral* create_interior_facet_integral(unsigned int i) const;

};

#ifndef UFC_BACKWARD_COMPATIBILITY 

// DOLFIN wrappers

// Standard library includes
#include <string>

// DOLFIN includes
#include <dolfin/common/NoDeleter.h>
#include <dolfin/fem/FiniteElement.h>
#include <dolfin/fem/DofMap.h>
#include <dolfin/fem/Form.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/function/GenericFunction.h>
#include <dolfin/function/CoefficientAssigner.h>
#include <dolfin/adaptivity/ErrorControl.h>
#include <dolfin/adaptivity/GoalFunctional.h>

namespace NSEErrEstGlobal
{

class CoefficientSpace_Rc: public dolfin::FunctionSpace
{
public:

  CoefficientSpace_Rc(const dolfin::Mesh& mesh):
    dolfin::FunctionSpace(dolfin::reference_to_no_delete_pointer(mesh),
                          const dolfin::FiniteElement* (new dolfin::FiniteElement(ufc::finite_element* (new nseerrestglobal_finite_element_0()))),
                          const dolfin::DofMap *(new dolfin::DofMap(ufc::dofmap* (new nseerrestglobal_dofmap_0()), mesh)))
  {
    // Do nothing
  }

  CoefficientSpace_Rc(dolfin::Mesh& mesh):
    dolfin::FunctionSpace(dolfin::reference_to_no_delete_pointer(mesh),
                          const dolfin::FiniteElement* (new dolfin::FiniteElement(ufc::finite_element* (new nseerrestglobal_finite_element_0()))),
                          const dolfin::DofMap* (new dolfin::DofMap(ufc::dofmap *(new nseerrestglobal_dofmap_0()), mesh)))
  {
    // Do nothing
  }

  CoefficientSpace_Rc(dolfin::Mesh*  mesh):
    dolfin::FunctionSpace(mesh,
                          const dolfin::FiniteElement *(new dolfin::FiniteElement(ufc::finite_element* (new nseerrestglobal_finite_element_0()))),
                          const dolfin::DofMap* (new dolfin::DofMap(ufc::dofmap*(new nseerrestglobal_dofmap_0()), *mesh)))
  {
      // Do nothing
  }

  CoefficientSpace_Rc(const dolfin::Mesh* mesh):
    dolfin::FunctionSpace(mesh,
                          const dolfin::FiniteElement *(new dolfin::FiniteElement(ufc::finite_element*(new nseerrestglobal_finite_element_0()))),
                          const dolfin::DofMap*(new dolfin::DofMap(ufc::dofmap*(new nseerrestglobal_dofmap_0()), *mesh)))
  {
      // Do nothing
  }

  ~CoefficientSpace_Rc()
  {
  }

};

class CoefficientSpace_Rm: public dolfin::FunctionSpace
{
public:

  CoefficientSpace_Rm(const dolfin::Mesh& mesh):
    dolfin::FunctionSpace(dolfin::reference_to_no_delete_pointer(mesh),
                          const dolfin::FiniteElement* (new dolfin::FiniteElement(ufc::finite_element* (new nseerrestglobal_finite_element_0()))),
                          const dolfin::DofMap *(new dolfin::DofMap(ufc::dofmap* (new nseerrestglobal_dofmap_0()), mesh)))
  {
    // Do nothing
  }

  CoefficientSpace_Rm(dolfin::Mesh& mesh):
    dolfin::FunctionSpace(dolfin::reference_to_no_delete_pointer(mesh),
                          const dolfin::FiniteElement* (new dolfin::FiniteElement(ufc::finite_element* (new nseerrestglobal_finite_element_0()))),
                          const dolfin::DofMap* (new dolfin::DofMap(ufc::dofmap *(new nseerrestglobal_dofmap_0()), mesh)))
  {
    // Do nothing
  }

  CoefficientSpace_Rm(dolfin::Mesh*  mesh):
    dolfin::FunctionSpace(mesh,
                          const dolfin::FiniteElement *(new dolfin::FiniteElement(ufc::finite_element* (new nseerrestglobal_finite_element_0()))),
                          const dolfin::DofMap* (new dolfin::DofMap(ufc::dofmap*(new nseerrestglobal_dofmap_0()), *mesh)))
  {
      // Do nothing
  }

  CoefficientSpace_Rm(const dolfin::Mesh* mesh):
    dolfin::FunctionSpace(mesh,
                          const dolfin::FiniteElement *(new dolfin::FiniteElement(ufc::finite_element*(new nseerrestglobal_finite_element_0()))),
                          const dolfin::DofMap*(new dolfin::DofMap(ufc::dofmap*(new nseerrestglobal_dofmap_0()), *mesh)))
  {
      // Do nothing
  }

  ~CoefficientSpace_Rm()
  {
  }

};

class CoefficientSpace_cv: public dolfin::FunctionSpace
{
public:

  CoefficientSpace_cv(const dolfin::Mesh& mesh):
    dolfin::FunctionSpace(dolfin::reference_to_no_delete_pointer(mesh),
                          const dolfin::FiniteElement* (new dolfin::FiniteElement(ufc::finite_element* (new nseerrestglobal_finite_element_0()))),
                          const dolfin::DofMap *(new dolfin::DofMap(ufc::dofmap* (new nseerrestglobal_dofmap_0()), mesh)))
  {
    // Do nothing
  }

  CoefficientSpace_cv(dolfin::Mesh& mesh):
    dolfin::FunctionSpace(dolfin::reference_to_no_delete_pointer(mesh),
                          const dolfin::FiniteElement* (new dolfin::FiniteElement(ufc::finite_element* (new nseerrestglobal_finite_element_0()))),
                          const dolfin::DofMap* (new dolfin::DofMap(ufc::dofmap *(new nseerrestglobal_dofmap_0()), mesh)))
  {
    // Do nothing
  }

  CoefficientSpace_cv(dolfin::Mesh*  mesh):
    dolfin::FunctionSpace(mesh,
                          const dolfin::FiniteElement *(new dolfin::FiniteElement(ufc::finite_element* (new nseerrestglobal_finite_element_0()))),
                          const dolfin::DofMap* (new dolfin::DofMap(ufc::dofmap*(new nseerrestglobal_dofmap_0()), *mesh)))
  {
      // Do nothing
  }

  CoefficientSpace_cv(const dolfin::Mesh* mesh):
    dolfin::FunctionSpace(mesh,
                          const dolfin::FiniteElement *(new dolfin::FiniteElement(ufc::finite_element*(new nseerrestglobal_finite_element_0()))),
                          const dolfin::DofMap*(new dolfin::DofMap(ufc::dofmap*(new nseerrestglobal_dofmap_0()), *mesh)))
  {
      // Do nothing
  }

  ~CoefficientSpace_cv()
  {
  }

};

class CoefficientSpace_h: public dolfin::FunctionSpace
{
public:

  CoefficientSpace_h(const dolfin::Mesh& mesh):
    dolfin::FunctionSpace(dolfin::reference_to_no_delete_pointer(mesh),
                          const dolfin::FiniteElement* (new dolfin::FiniteElement(ufc::finite_element* (new nseerrestglobal_finite_element_0()))),
                          const dolfin::DofMap *(new dolfin::DofMap(ufc::dofmap* (new nseerrestglobal_dofmap_0()), mesh)))
  {
    // Do nothing
  }

  CoefficientSpace_h(dolfin::Mesh& mesh):
    dolfin::FunctionSpace(dolfin::reference_to_no_delete_pointer(mesh),
                          const dolfin::FiniteElement* (new dolfin::FiniteElement(ufc::finite_element* (new nseerrestglobal_finite_element_0()))),
                          const dolfin::DofMap* (new dolfin::DofMap(ufc::dofmap *(new nseerrestglobal_dofmap_0()), mesh)))
  {
    // Do nothing
  }

  CoefficientSpace_h(dolfin::Mesh*  mesh):
    dolfin::FunctionSpace(mesh,
                          const dolfin::FiniteElement *(new dolfin::FiniteElement(ufc::finite_element* (new nseerrestglobal_finite_element_0()))),
                          const dolfin::DofMap* (new dolfin::DofMap(ufc::dofmap*(new nseerrestglobal_dofmap_0()), *mesh)))
  {
      // Do nothing
  }

  CoefficientSpace_h(const dolfin::Mesh* mesh):
    dolfin::FunctionSpace(mesh,
                          const dolfin::FiniteElement *(new dolfin::FiniteElement(ufc::finite_element*(new nseerrestglobal_finite_element_0()))),
                          const dolfin::DofMap*(new dolfin::DofMap(ufc::dofmap*(new nseerrestglobal_dofmap_0()), *mesh)))
  {
      // Do nothing
  }

  ~CoefficientSpace_h()
  {
  }

};

class CoefficientSpace_wc: public dolfin::FunctionSpace
{
public:

  CoefficientSpace_wc(const dolfin::Mesh& mesh):
    dolfin::FunctionSpace(dolfin::reference_to_no_delete_pointer(mesh),
                          const dolfin::FiniteElement* (new dolfin::FiniteElement(ufc::finite_element* (new nseerrestglobal_finite_element_0()))),
                          const dolfin::DofMap *(new dolfin::DofMap(ufc::dofmap* (new nseerrestglobal_dofmap_0()), mesh)))
  {
    // Do nothing
  }

  CoefficientSpace_wc(dolfin::Mesh& mesh):
    dolfin::FunctionSpace(dolfin::reference_to_no_delete_pointer(mesh),
                          const dolfin::FiniteElement* (new dolfin::FiniteElement(ufc::finite_element* (new nseerrestglobal_finite_element_0()))),
                          const dolfin::DofMap* (new dolfin::DofMap(ufc::dofmap *(new nseerrestglobal_dofmap_0()), mesh)))
  {
    // Do nothing
  }

  CoefficientSpace_wc(dolfin::Mesh*  mesh):
    dolfin::FunctionSpace(mesh,
                          const dolfin::FiniteElement *(new dolfin::FiniteElement(ufc::finite_element* (new nseerrestglobal_finite_element_0()))),
                          const dolfin::DofMap* (new dolfin::DofMap(ufc::dofmap*(new nseerrestglobal_dofmap_0()), *mesh)))
  {
      // Do nothing
  }

  CoefficientSpace_wc(const dolfin::Mesh* mesh):
    dolfin::FunctionSpace(mesh,
                          const dolfin::FiniteElement *(new dolfin::FiniteElement(ufc::finite_element*(new nseerrestglobal_finite_element_0()))),
                          const dolfin::DofMap*(new dolfin::DofMap(ufc::dofmap*(new nseerrestglobal_dofmap_0()), *mesh)))
  {
      // Do nothing
  }

  ~CoefficientSpace_wc()
  {
  }

};

class CoefficientSpace_wm: public dolfin::FunctionSpace
{
public:

  CoefficientSpace_wm(const dolfin::Mesh& mesh):
    dolfin::FunctionSpace(dolfin::reference_to_no_delete_pointer(mesh),
                          const dolfin::FiniteElement* (new dolfin::FiniteElement(ufc::finite_element* (new nseerrestglobal_finite_element_0()))),
                          const dolfin::DofMap *(new dolfin::DofMap(ufc::dofmap* (new nseerrestglobal_dofmap_0()), mesh)))
  {
    // Do nothing
  }

  CoefficientSpace_wm(dolfin::Mesh& mesh):
    dolfin::FunctionSpace(dolfin::reference_to_no_delete_pointer(mesh),
                          const dolfin::FiniteElement* (new dolfin::FiniteElement(ufc::finite_element* (new nseerrestglobal_finite_element_0()))),
                          const dolfin::DofMap* (new dolfin::DofMap(ufc::dofmap *(new nseerrestglobal_dofmap_0()), mesh)))
  {
    // Do nothing
  }

  CoefficientSpace_wm(dolfin::Mesh*  mesh):
    dolfin::FunctionSpace(mesh,
                          const dolfin::FiniteElement *(new dolfin::FiniteElement(ufc::finite_element* (new nseerrestglobal_finite_element_0()))),
                          const dolfin::DofMap* (new dolfin::DofMap(ufc::dofmap*(new nseerrestglobal_dofmap_0()), *mesh)))
  {
      // Do nothing
  }

  CoefficientSpace_wm(const dolfin::Mesh* mesh):
    dolfin::FunctionSpace(mesh,
                          const dolfin::FiniteElement *(new dolfin::FiniteElement(ufc::finite_element*(new nseerrestglobal_finite_element_0()))),
                          const dolfin::DofMap*(new dolfin::DofMap(ufc::dofmap*(new nseerrestglobal_dofmap_0()), *mesh)))
  {
      // Do nothing
  }

  ~CoefficientSpace_wm()
  {
  }

};

typedef CoefficientSpace_h Form_0_FunctionSpace_0;

typedef CoefficientSpace_cv Form_0_FunctionSpace_1;

typedef CoefficientSpace_Rm Form_0_FunctionSpace_2;

typedef CoefficientSpace_Rc Form_0_FunctionSpace_3;

typedef CoefficientSpace_wm Form_0_FunctionSpace_4;

typedef CoefficientSpace_wc Form_0_FunctionSpace_5;

class Form_0: public dolfin::Form
{
public:

  // Constructor
  Form_0(const dolfin::Mesh& mesh):
    dolfin::Form(0, 6), h(*this, 0), cv(*this, 1), Rm(*this, 2), Rc(*this, 3), wm(*this, 4), wc(*this, 5)
  {
    _mesh = reference_to_no_delete_pointer(mesh);
    _ufc_form = const ufc::form* (new nseerrestglobal_form_0());
  }

  // Constructor
  Form_0(const dolfin::Mesh& mesh, const dolfin::GenericFunction& h, const dolfin::GenericFunction& cv, const dolfin::GenericFunction& Rm, const dolfin::GenericFunction& Rc, const dolfin::GenericFunction& wm, const dolfin::GenericFunction& wc):
    dolfin::Form(0, 6), h(*this, 0), cv(*this, 1), Rm(*this, 2), Rc(*this, 3), wm(*this, 4), wc(*this, 5)
  {
    _mesh = reference_to_no_delete_pointer(mesh);
    this->h = h;
    this->cv = cv;
    this->Rm = Rm;
    this->Rc = Rc;
    this->wm = wm;
    this->wc = wc;

    _ufc_form = const ufc::form* (new nseerrestglobal_form_0());
  }

  // Constructor
  Form_0(const dolfin::Mesh& mesh, boost::shared_ptr<const dolfin::GenericFunction> h, boost::shared_ptr<const dolfin::GenericFunction> cv, boost::shared_ptr<const dolfin::GenericFunction> Rm, boost::shared_ptr<const dolfin::GenericFunction> Rc, boost::shared_ptr<const dolfin::GenericFunction> wm, boost::shared_ptr<const dolfin::GenericFunction> wc):
    dolfin::Form(0, 6), h(*this, 0), cv(*this, 1), Rm(*this, 2), Rc(*this, 3), wm(*this, 4), wc(*this, 5)
  {
    _mesh = reference_to_no_delete_pointer(mesh);
    this->h = *h;
    this->cv = *cv;
    this->Rm = *Rm;
    this->Rc = *Rc;
    this->wm = *wm;
    this->wc = *wc;

    _ufc_form = const ufc::form* (new nseerrestglobal_form_0());
  }

  // Constructor
  Form_0(boost::shared_ptr<const dolfin::Mesh> mesh):
    dolfin::Form(0, 6), h(*this, 0), cv(*this, 1), Rm(*this, 2), Rc(*this, 3), wm(*this, 4), wc(*this, 5)
  {
    _mesh = mesh;
    _ufc_form = const ufc::form* (new nseerrestglobal_form_0());
  }

  // Constructor
  Form_0(boost::shared_ptr<const dolfin::Mesh> mesh, const dolfin::GenericFunction& h, const dolfin::GenericFunction& cv, const dolfin::GenericFunction& Rm, const dolfin::GenericFunction& Rc, const dolfin::GenericFunction& wm, const dolfin::GenericFunction& wc):
    dolfin::Form(0, 6), h(*this, 0), cv(*this, 1), Rm(*this, 2), Rc(*this, 3), wm(*this, 4), wc(*this, 5)
  {
    _mesh = mesh;
    this->h = h;
    this->cv = cv;
    this->Rm = Rm;
    this->Rc = Rc;
    this->wm = wm;
    this->wc = wc;

    _ufc_form = const ufc::form* (new nseerrestglobal_form_0());
  }

  // Constructor
  Form_0(boost::shared_ptr<const dolfin::Mesh> mesh, boost::shared_ptr<const dolfin::GenericFunction> h, boost::shared_ptr<const dolfin::GenericFunction> cv, boost::shared_ptr<const dolfin::GenericFunction> Rm, boost::shared_ptr<const dolfin::GenericFunction> Rc, boost::shared_ptr<const dolfin::GenericFunction> wm, boost::shared_ptr<const dolfin::GenericFunction> wc):
    dolfin::Form(0, 6), h(*this, 0), cv(*this, 1), Rm(*this, 2), Rc(*this, 3), wm(*this, 4), wc(*this, 5)
  {
    _mesh = mesh;
    this->h = *h;
    this->cv = *cv;
    this->Rm = *Rm;
    this->Rc = *Rc;
    this->wm = *wm;
    this->wc = *wc;

    _ufc_form = const ufc::form* (new nseerrestglobal_form_0());
  }

  // Destructor
  ~Form_0()
  {}

  /// Return the number of the coefficient with this name
  virtual dolfin::uint coefficient_number(const std::string& name) const
  {
    if (name == "h")
      return 0;
    else if (name == "cv")
      return 1;
    else if (name == "Rm")
      return 2;
    else if (name == "Rc")
      return 3;
    else if (name == "wm")
      return 4;
    else if (name == "wc")
      return 5;

    dolfin::dolfin_error("generated code for class Form",
                         "access coeficient data",
                         "Invalid coeficient");
    return 0;
  }

  /// Return the name of the coefficient with this number
  virtual std::string coefficient_name(dolfin::uint i) const
  {
    switch (i)
    {
    case 0:
      return "h";
    case 1:
      return "cv";
    case 2:
      return "Rm";
    case 3:
      return "Rc";
    case 4:
      return "wm";
    case 5:
      return "wc";
    }

    dolfin::dolfin_error("generated code for class Form",
                         "access coeficient data",
                         "Invalid coeficient");
    return "unnamed";
  }

  // Typedefs
  typedef Form_0_FunctionSpace_0 CoefficientSpace_h;
  typedef Form_0_FunctionSpace_1 CoefficientSpace_cv;
  typedef Form_0_FunctionSpace_2 CoefficientSpace_Rm;
  typedef Form_0_FunctionSpace_3 CoefficientSpace_Rc;
  typedef Form_0_FunctionSpace_4 CoefficientSpace_wm;
  typedef Form_0_FunctionSpace_5 CoefficientSpace_wc;

  // Coefficients
  dolfin::CoefficientAssigner h;
  dolfin::CoefficientAssigner cv;
  dolfin::CoefficientAssigner Rm;
  dolfin::CoefficientAssigner Rc;
  dolfin::CoefficientAssigner wm;
  dolfin::CoefficientAssigner wc;
};

// Class typedefs
typedef Form_0 Functional;

}
#else 

// DOLFIN wrappers
#include <dolfin/fem/Form.h>

class NSEErrEstGlobalFunctional : public dolfin::Form
{
public:

  NSEErrEstGlobalFunctional(dolfin::Function& w0, dolfin::Function& w1, dolfin::Function& w2, dolfin::Function& w3, dolfin::Function& w4, dolfin::Function& w5) : dolfin::Form()
  {
    __coefficients.push_back(&w0);
    __coefficients.push_back(&w1);
    __coefficients.push_back(&w2);
    __coefficients.push_back(&w3);
    __coefficients.push_back(&w4);
    __coefficients.push_back(&w5);
  }

  /// Return UFC form
  virtual const ufc::form& form() const
  {
    return __form;
  }
  
  /// Return array of coefficients
  virtual const dolfin::Array<dolfin::Function*>& coefficients() const
  {
    return __coefficients;
  }

private:

  // UFC form
  nseerrestglobal_form_0 __form;

  /// Array of coefficients
  dolfin::Array<dolfin::Function*> __coefficients;

};


#endif 


#endif