/**
 * Variational Inference MPC (VI-MPC) controller based on
 * Okada & Taniguchi, "Variational Inference MPC for Bayesian Model-based Reinforcement Learning",
 * arXiv:1907.04202.
 *
 * Relative to Vanilla MPPI, VI-MPC moment-matches both the mean μ and diagonal covariance Σ of
 * the variational action distribution from importance weights. The Σ update is implemented in
 * the sampling distribution when SamplingParams::update_variance_from_weights is enabled; this
 * controller reuses VanillaMPPIController::computeControl() for the iterative optimization loop.
 **/

#ifndef MPPIGENERIC_VARIATIONAL_MPPI_CONTROLLER_CUH
#define MPPIGENERIC_VARIATIONAL_MPPI_CONTROLLER_CUH

#include <mppi/controllers/MPPI/mppi_controller.cuh>

template <class DYN_T, class COST_T, class FB_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS,
          class SAMPLING_T = ::mppi::sampling_distributions::GaussianDistribution<typename DYN_T::DYN_PARAMS_T>,
          class PARAMS_T = ControllerParams<DYN_T::STATE_DIM, DYN_T::CONTROL_DIM, MAX_TIMESTEPS>>
class VariationalMPPIController
  : public VanillaMPPIController<DYN_T, COST_T, FB_T, MAX_TIMESTEPS, NUM_ROLLOUTS, SAMPLING_T, PARAMS_T>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  typedef VanillaMPPIController<DYN_T, COST_T, FB_T, MAX_TIMESTEPS, NUM_ROLLOUTS, SAMPLING_T, PARAMS_T> PARENT_CLASS;
  using control_trajectory = typename PARENT_CLASS::control_trajectory;

  VariationalMPPIController(DYN_T* model, COST_T* cost, FB_T* fb_controller, SAMPLING_T* sampler, float dt, int max_iter,
                            float lambda, float alpha, int num_timesteps = MAX_TIMESTEPS,
                            const Eigen::Ref<const control_trajectory>& init_control_traj = control_trajectory::Zero(),
                            cudaStream_t stream = nullptr);

  VariationalMPPIController(DYN_T* model, COST_T* cost, FB_T* fb_controller, SAMPLING_T* sampler, PARAMS_T& params,
                            cudaStream_t stream = nullptr);

  ~VariationalMPPIController() = default;

  std::string getControllerName() const override
  {
    return "Variational MPPI";
  }
};

#if __CUDACC__
#include "variational_mppi_controller.cu"
#endif

#endif  // MPPIGENERIC_VARIATIONAL_MPPI_CONTROLLER_CUH
