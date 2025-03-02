#ifndef REACTORRK54_H
#define REACTORRK54_H

#include "ReactorBase.H"

namespace pele::physics::reactions {

// stages and coefficients for low storage rk54
struct RK54Params
{
  const int nstages_rk54 = 5;
  const amrex::Real alpha_rk54[5] = {
    0.225022458725713,
    0.544043312951405,
    0.144568243493995,
    0.786664342198357,
    0.348667178999280
  };

  const amrex::Real beta_rk54[5] = {
    -0.173793152085374,
    -0.163088487225003,
    -0.517920839886378,
    -0.194163057171995,
    0.0
  };
  

  const amrex::Real err_rk54[5] = {
    -0.085988015462880,
    0.189074063397015,
    -0.144145875232851,
    -0.031793391517533,
    0.072853218816250
  };
  const amrex::Real exp1_rk54 = 0.25;
  const amrex::Real exp2_rk54 = 0.2;
  const amrex::Real betaerr_rk54 = 1.0;
};

class ReactorRK54 : public ReactorBase::Register<ReactorRK54>
{
public:
  static std::string identifier() { return "ReactorRK54"; }

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
  int rk54_nsubsteps_guess{10};
  int rk54_nsubsteps_min{5};
  int rk54_nsubsteps_max{500};
  utils::FlattenOps<Ordering> flatten_ops;
  int m_reactor_type{0};
  bool m_clean_init_massfrac{false};
};
} // namespace pele::physics::reactions
#endif
