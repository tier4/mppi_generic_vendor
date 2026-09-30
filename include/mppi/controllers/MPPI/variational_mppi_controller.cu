#include <mppi/controllers/MPPI/variational_mppi_controller.cuh>

#define VARIATIONAL_MPPI_TEMPLATE                                                                                      \
  template <class DYN_T, class COST_T, class FB_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS, class SAMPLING_T,              \
            class PARAMS_T>

#define VariationalMPPI                                                                                                \
  VariationalMPPIController<DYN_T, COST_T, FB_T, MAX_TIMESTEPS, NUM_ROLLOUTS, SAMPLING_T, PARAMS_T>

VARIATIONAL_MPPI_TEMPLATE
VariationalMPPI::VariationalMPPIController(DYN_T* model, COST_T* cost, FB_T* fb_controller, SAMPLING_T* sampler, float dt,
                                           int max_iter, float lambda, float alpha, int num_timesteps,
                                           const Eigen::Ref<const control_trajectory>& init_control_traj,
                                           cudaStream_t stream)
  : PARENT_CLASS(model, cost, fb_controller, sampler, dt, max_iter, lambda, alpha, num_timesteps, init_control_traj,
                 stream)
{
}

VARIATIONAL_MPPI_TEMPLATE
VariationalMPPI::VariationalMPPIController(DYN_T* model, COST_T* cost, FB_T* fb_controller, SAMPLING_T* sampler,
                                           PARAMS_T& params, cudaStream_t stream)
  : PARENT_CLASS(model, cost, fb_controller, sampler, params, stream)
{
}

#undef VARIATIONAL_MPPI_TEMPLATE
#undef VariationalMPPI
