#ifndef REACTORRK43_H
#define REACTORRK43_H

#include "ReactorBase.H"

namespace pele::physics::reactions {

// stages and coefficients for low storage rk43
struct RK43Params
{
  const int nstages_rk43 = 4;
  const amrex::Real alpha_rk43[4] = {
    0.324165738828746,
    0.557097864505543,
    -0.086054914312728,
    -2.681109033041384
  };

  const amrex::Real beta_rk43[4] = {
    -0.220085869553644,
    0.044841272376718,
    3.061144941196749,
    0.0    
  };
  
  const amrex::Real err_rk43[4] = {
    -0.236601614805741,
    0.511023906795933,
    0.108593284158578,
    -0.383015576148769
  };
  const amrex::Real exp1_rk43 = 0.25;
  const amrex::Real exp2_rk43 = 0.2;
  const amrex::Real betaerr_rk43 = 1.0;
};

class ReactorRK43 : public ReactorBase::Register<ReactorRK43>
{
public:
  static std::string identifier() { return "ReactorRK43"; }

  using Ordering = utils::YCOrder;

  int init(int reactor_type, int ncells) override;

  void close() override {}

  void print_final_stats(void* /*mem*/) override {}

  int react(
    const amrex::Box& box,
    amrex::Array4<amrex::Real> const& rY_in,
    amrex::Array4<amrex::Real> const& rYsrc_in,
    amrex::Array4<amrex::Real> const& T_in,
    amrex::Array4<amrex::Real> const& rEner_in,
    amrex::Array4<amrex::Real> const& rEner_src_in,
    amrex::Array4<amrex::Real> const& FC_in,
    amrex::Array4<int> const& mask,
    amrex::Real& dt_react,
    amrex::Real& time
#ifdef AMREX_USE_GPU
    ,
    amrex::gpuStream_t stream
#endif
    ) override;

  int react(
    amrex::Real* rY_in,
    amrex::Real* rYsrc_in,
    amrex::Real* rX_in,
    amrex::Real* rX_src_in,
    amrex::Real& dt_react,
    amrex::Real& time,
    int ncells
#ifdef AMREX_USE_GPU
    ,
    amrex::gpuStream_t stream
#endif
    ) override;

  void flatten(
    const amrex::Box& box,
    const int ncells,
    amrex::Array4<const amrex::Real> const& rhoY,
    amrex::Array4<const amrex::Real> const& frcExt,
    amrex::Array4<const amrex::Real> const& temperature,
    amrex::Array4<const amrex::Real> const& rhoE,
    amrex::Array4<const amrex::Real> const& frcEExt,
    amrex::Real* y_vect,
    amrex::Real* src_vect,
    amrex::Real* vect_energy,
    amrex::Real* src_vect_energy) override
  {
    flatten_ops.flatten(
      box, ncells, m_reactor_type, m_clean_init_massfrac, rhoY, frcExt,
      temperature, rhoE, frcEExt, y_vect, src_vect, vect_energy,
      src_vect_energy);
  }

  void unflatten(
    const amrex::Box& box,
    const int ncells,
    amrex::Array4<amrex::Real> const& rhoY,
    amrex::Array4<amrex::Real> const& temperature,
    amrex::Array4<amrex::Real> const& rhoE,
    amrex::Array4<amrex::Real> const& frcEExt,
    amrex::Array4<amrex::Real> const& FC_in,
    amrex::Real* y_vect,
    amrex::Real* vect_energy,
    long int* FCunt,
    amrex::Real dt) override
  {
    flatten_ops.unflatten(
      box, ncells, m_reactor_type, m_clean_init_massfrac, rhoY, temperature,
      rhoE, frcEExt, FC_in, y_vect, vect_energy, FCunt, dt);
  }

  void set_eos_parm(
    const pele::physics::eos::EosParm<pele::physics::PhysicsType::eos_type>*
      h_eosparm,
    const pele::physics::eos::EosParm<pele::physics::PhysicsType::eos_type>*
      d_eosparm) override
  {
    m_h_eosparm = h_eosparm;
    m_d_eosparm = d_eosparm;
  }

private:
  amrex::Real absTol{1e-10};
  int rk43_nsubsteps_guess{10};
  int rk43_nsubsteps_min{5};
  int rk43_nsubsteps_max{50};
  utils::FlattenOps<Ordering> flatten_ops;
  int m_reactor_type{0};
  bool m_clean_init_massfrac{false};
};
} // namespace pele::physics::reactions
#endif
