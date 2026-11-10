// Pinned LQR/bicycle arithmetic, float64, no fast-math, no prediction cache.
// Velocity/curvature least-squares profiles remain the native NumPy routine.
#include <algorithm>
#include <cmath>
#include <cstring>

static double angle(double value) { return std::atan2(std::sin(value), std::cos(value)); }
static double principal(double value) {
    const double pi = std::acos(-1.0);
    double r = std::fmod(value + pi, 2.0 * pi);
    if (r < 0) r += 2.0 * pi;
    return r - pi;
}

extern "C" int coordiworld_controller(int batch, int steps, int horizon,
    const double* proposals, const double* velocity, const double* curvature,
    const double* initial, const double* parameters, double* output) {
    if (batch < 1 || steps < 2 || horizon < 2) return 1;
    const double dt=parameters[0], wb=parameters[1], motion_wb=parameters[2];
    const double atc=parameters[3], stc=parameters[4], max_steer=parameters[5];
    const double qlong=parameters[6], rlong=parameters[7];
    const double q0=parameters[8], q1=parameters[9], q2=parameters[10];
    const double rlat=parameters[11], stop_gain=parameters[12], stop_velocity=parameters[13];
    for (int j=0; j<batch; ++j) {
        double* base=output+j*(steps+1)*11;
        std::memcpy(base, initial, sizeof(double)*11);
        for (int t=0; t<steps; ++t) {
            const double* old=base+t*11;
            double* next=base+(t+1)*11;
            const double* target=proposals+(j*(steps+1)+t)*11;
            const int ref=std::min(t+horizon, steps-1);
            const double reference_velocity=velocity[j*steps+ref];
            double acceleration, rate=0;
            if (reference_velocity<=stop_velocity && old[3]<=stop_velocity) {
                acceleration=-stop_gain*(old[3]-reference_velocity);
            } else {
                const double B=horizon*dt;
                acceleration=(-1.0/(B*qlong*B+rlong))*B*qlong*(old[3]-reference_velocity);
                const double dx=old[0]-target[0], dy=old[1]-target[1];
                const double e0=-dx*std::sin(target[2])+dy*std::cos(target[2]);
                const double e1=angle(old[2]-target[2]), e2=old[7];
                double A01=0, A12=0, A02=0, B0=0, B1=0, g0=0, g1=0;
                double prefix_b=0, prefix_b_input=0, prefix_c=0, increment=0;
                for (int k=0; k<horizon; ++k) {
                    const double v=old[3]+increment;
                    const double a=v*dt, b=a/wb;
                    const double c=-v*curvature[j*steps+std::min(t+k,ref)]*dt;
                    const double bi=b*(dt*k);
                    A01+=a; A12+=b; A02+=a*prefix_b;
                    B0+=a*prefix_b_input; B1+=bi;
                    g0+=a*prefix_c; g1+=c;
                    prefix_b+=b; prefix_b_input+=bi; prefix_c+=c;
                    increment+=acceleration*dt;
                }
                const double B2=horizon*dt;
                const double z0=(e0+A01*e1+A02*e2)+g0;
                const double z1=angle((e1+A12*e2)+g1);
                const double z2=angle(e2);
                const double denominator=(B0*q0)*B0+(B1*q1)*B1+(B2*q2)*B2+rlat;
                const double tail=(B0*q0)*z0+(B1*q1)*z1+(B2*q2)*z2;
                rate=(-1.0/denominator)*tail;
            }
            const double filtered_accel=dt/(dt+atc)*(acceleration-old[5])+old[5];
            const double ideal_steering=dt*rate+old[7];
            const double filtered_steering=dt/(dt+stc)*(ideal_steering-old[7])+old[7];
            const double filtered_rate=(filtered_steering-old[7])/dt;
            std::memcpy(next,old,sizeof(double)*11);
            next[0]=old[0]+(old[3]*std::cos(old[2]))*dt;
            next[1]=old[1]+(old[3]*std::sin(old[2]))*dt;
            next[2]=principal(old[2]+(old[3]*std::tan(old[7])/motion_wb)*dt);
            next[3]=old[3]+filtered_accel*dt; next[4]=0;
            next[5]=filtered_accel; next[6]=0;
            next[7]=std::clamp(old[7]+filtered_rate*dt,-max_steer,max_steer);
            next[8]=filtered_rate;
            next[9]=next[3]*std::tan(next[7])/motion_wb;
            next[10]=(next[9]-old[9])/dt;
            for (int k=0;k<11;++k) if (!std::isfinite(next[k])) return 2;
        }
    }
    return 0;
}
