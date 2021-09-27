#define N_MAX 4
#define M_PI 3.14159265358979323846
#define NUM_THREADS 10

// #define EX37

#ifdef EX31 // nonlocal pde
#define MC_SAMPLES 10
#define eq_name "Example 3.1"
#define rdim 1
#define g(x) ArrayXd tmp = ArrayXd::Zero(1, 1); tmp(0) = exp(-x.square().sum())
#define X_sde(s, t, x, w) w = (x - 0.1 * sqrt(2. * (t - s)) * w);
#define shift_me(a, b) a
#define fn(x, xx, y, yy,d) ArrayXd ret = ArrayXd::Zero(1, 1); ret(0) = - yy(0)
#endif

#ifdef EX41 // nonlocal pde
#define MC_SAMPLES 10
#define eq_name "Example 4.1"
#define rdim 1
#define g(x) ArrayXd tmp = ArrayXd::Zero(1, 1); tmp(0) = exp(-x.square().sum())
#define X_sde(s, t, x, w) w = (x - 0.1 * sqrt(2. * (t - s)) * w);
#define shift_me(a, b) a
#define fn(x, xx, y, yy,d) ArrayXd ret = ArrayXd::Zero(1, 1); ret(0) = y(0)*(1 - 0.1 * yy(0))
#endif

#ifdef EXnonlocal_allencahn // nonlocal pde
#define MC_SAMPLES 10
#define eq_name "Example Sine Gordon"
#define rdim 1
#define g(x) ArrayXd tmp = ArrayXd::Zero(1, 1); tmp(0) = exp(-x.square().sum())
#define X_sde(s, t, x, w) w = (x - 0.1 * sqrt(2. * (t - s)) * w);
#define shift_me(a, b) a
#define fn(x, xx, y, yy,d) ArrayXd ret = ArrayXd::Zero(1, 1); ret(0) = y(0)*(1 - 0.1 * (yy(0) - pow(yy(0),3)))
#endif

#ifdef EX32 // reflected non local pde
#define MC_SAMPLES 10
#define eq_name "Example 3.2"
#define rdim 1 // result dimension
#define g(x) ArrayXd tmp = ArrayXd::Zero(1, 1); tmp(0) = exp(-x.square().sum()) // final conditions
#define X_sde(s, t, x, w) w = (x + sqrt(2. * (t - s)) * w); reflect(x, w, d, 0, 1); // SDE reflected
#define shift_me(a, b) a
#define fn(x, xx, y, yy,d) ArrayXd ret = ArrayXd::Zero(1, 1); ret(0) = - yy(0) // non local function
#endif

#ifdef EX33 // non local sine gordon
#define MC_SAMPLES 10
#define eq_name "Example 3.3"
#define rdim 1
#define g(x) ArrayXd tmp = ArrayXd::Zero(1, 1); tmp(0) = exp(-x.square().sum())
#define X_sde(s, t, x, w) w = (x + sqrt(2. * (t - s)) * w);
#define shift_me(a, b) a
#define fn(x, xx, y, yy,d) ArrayXd ret = ArrayXd::Zero(1, 1); ret(0) = - yy(0) + sin(y(0))
#endif

#ifdef EXsinegordon // non local sine gordon, shifted integration
#define MC_SAMPLES 10
#define eq_name "Example Non local sine gordon"
#define rdim 1
#define g(x) ArrayXd tmp = ArrayXd::Zero(1, 1); tmp(0) = exp(- 0.5 * x.square().sum())
#define X_sde(s, t, x, w) w = (x + 0.1 * sqrt((t - s)) * w);
#define shift_me(a, b) a * 0.1  / sqrt(2.) + b
// todo: change 10 to d
#define fn(x, xx, y, yy,d) ArrayXd ret = ArrayXd::Zero(1, 1); ret(0) = (sin(y(0)) - yy(0) * pow(double(M_PI), d /2.) * pow(0.1,d) ) 
#endif

#ifdef EX34b // non local sine gordon, shifted integration
#define MC_SAMPLES 10
#define eq_name "Example 3.4bis"
#define rdim 1
#define g(x) ArrayXd tmp = ArrayXd::Zero(1, 1); tmp(0) = exp(-x.square().sum())
#define X_sde(s, t, x, w) w = (x - 0.1 * sqrt(2. * (t - s)) * w);
#define shift_me(a, b) a / sqrt(2.) + b(0)
#define fn(x, xx, y, yy,d) ArrayXd ret = ArrayXd::Zero(1, 1); ret(0) =  sin(y(0)) - yy(0)
#endif

#ifdef EXNLFisherKPP // non local fisher kpp, shifted integration
#define MC_SAMPLES 10
#define eq_name "Example NLFisherKPP"
#define rdim 1
#define g(x) ArrayXd tmp = ArrayXd::Zero(1, 1); tmp(0) = exp(- 0.5 * x.square().sum())
#define X_sde(s, t, x, w) w = (x - 0.1 * sqrt((t - s)) * w);
#define shift_me(a, b) b + a * 0.1 / sqrt(2.) //* ArrayXd::Ones(d) //a * 0.1 / sqrt(2.) //+ b(0) /// 
// todo: change 10 to d
#define fn(x, xx, y, yy, d) ArrayXd ret = ArrayXd::Zero(1, 1); ret(0) = std::max(0.,y(0)) * (1. -  std::max(0.,yy(0)) * pow(double(M_PI), d /2.) * pow(0.1,d))
// * pow(double(M_PI),1. / 2.) * 0.1
#endif

#ifdef EXFisherKPP_r // non local fisher kpp, shifted integration
#define MC_SAMPLES 1
#define eq_name "Example FisherKPP reflected"
#define rdim 1
#define g(x) ArrayXd tmp = ArrayXd::Zero(1, 1); tmp(0) = exp(- 0.5 * x.square().sum())
#define X_sde(s, t, x, w) w = (x - sqrt((t - s)) * w);reflect(x, w, d, -0.1, 0.1);
#define shift_me(a, b) b + a * 0.1 / sqrt(2.) //* ArrayXd::Ones(d) //a * 0.1 / sqrt(2.) //+ b(0) /// 
// todo: change 10 to d
#define fn(x, xx, y, yy, d) ArrayXd ret = ArrayXd::Zero(1, 1); ret(0) = std::max(0.,y(0)) * (1. -  std::max(0.,y(0))) //* pow(double(M_PI), d /2.) * pow(0.1,d))
// * pow(double(M_PI),1. / 2.) * 0.1
#endif

#ifdef EX36 // reflected fisher kpp, shifted integration,reflected
#define MC_SAMPLES 10
#define eq_name "Example 3.6"
#define rdim 1
#define g(x) ArrayXd tmp = ArrayXd::Zero(1, 1); tmp(0) = exp(-x.square().sum())
#define X_sde(s, t, x, w) w = (x - sqrt(2. * (t - s)) * w); reflect(x, w, d, 0, 1);
#define shift_me(a, b) a
// todo: change 10 to d
#define fn(x, xx, y, yy,d) ArrayXd ret = ArrayXd::Zero(1, 1); ret(0) = y(0)*(1-y(0))
#endif

#ifdef EX37 // mirrahimi example
#define MC_SAMPLES 10
#define eq_name "Example 3.7"
#define rdim 1
#define g(x) ArrayXd tmp = ArrayXd::Zero(1, 1); tmp(0) = exp( -0.5 * x.square().sum())
#define X_sde(s, t, x, w) w = (x - 0.1 * sqrt(2.  *(t - s)) * w );
#define shift_me(a, b) a
// todo: change 10 to d
#define fn(x, xx, y, yy,d) ArrayXd ret = ArrayXd::Zero(1, 1); ret(0) = y(0) * ( 1. - 0.1 * x.square().sum() - 0.1 * yy(0) * pow(2. * double(M_PI), double(x.size()) / 2.) * exp( 0.5 * xx.square().sum()) )
#endif

#ifdef EXhamel // mutator replicator
#define MC_SAMPLES 10
#define eq_name "Example 3.8"
#define rdim 1
#define g(x) ArrayXd tmp = ArrayXd::Zero(1, 1); tmp(0) = pow(2, double(x.size()/2.)) * exp( - 2. * double(M_PI) * x.square().sum())
#define X_sde(s, t, x, w) w = (x - 0.1 * sqrt( (t - s)) * w );
#define shift_me(a, b) a
// todo: change 10 to d
#define fn(x, xx, y, yy,d) ArrayXd ret = ArrayXd::Zero(1, 1); ret(0) = std::max(0.,y(0)) * ( - 0.5 * x.square().sum() + std::max(0.,yy(0)) * 0.5 * xx.square().sum() * pow(2. * double(M_PI), d / 2.) * exp( 0.5 * xx.square().sum()  ) )
#endif

#ifdef EX310 // quadratic birth, non local competition
#define MC_SAMPLES 10
#define eq_name "Example 3.10"
#define rdim 1
#define g(x) ArrayXd tmp = ArrayXd::Zero(1, 1); tmp(0) =  exp( - 0.5 * x.square().sum()) // / pow(2. * double(M_PI), 10./2.)
#define X_sde(s, t, x, w) w = (x - 0.1 * sqrt( 2. * (t - s)) * w );
#define shift_me(a, b) a + b(0)
// todo: change 10 to d
#define fn(x, xx, y, yy,d) ArrayXd ret = ArrayXd::Zero(1, 1); ret(0) = std::max(0. ,y(0)) * ( 1. - x.square().sum() - std::max(0.,yy(0)) )
#endif

#ifdef EX310b // gaussian birth, non local competition
#define MC_SAMPLES 10
#define eq_name "Example 3.10b"
#define rdim 1
#define g(x) ArrayXd tmp = ArrayXd::Zero(1, 1); tmp(0) =  exp( - 0.5 * x.square().sum()) / pow(2. * double(M_PI), 1./2.)
#define X_sde(s, t, x, w) w = (x - 0.1 * sqrt( 2. * (t - s)) * w );
#define shift_me(a, b) a + b(0)
// todo: change 10 to d
#define fn(x, xx, y, yy,d) ArrayXd ret = ArrayXd::Zero(1, 1); ret(0) = y(0) * ( exp(- 0.5 * x.square().sum()) - yy(0) )
#endif

#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstdint>
#include <random>
#include <cmath>
#include <ctime>
#include <thread>
#include <chrono>
#include <Eigen/Dense>

using Eigen::ArrayXd;
using Eigen::VectorXd;

struct mlp_t {
	uint8_t m; // number of samples for multi level monte carlo
	uint8_t n; // number of samples for monte carlo integration of non local term
	uint8_t thread_num;
	uint16_t d; // dimension
	ArrayXd x;
	double s;
	double t;
	ArrayXd res;
};


ArrayXd f(const ArrayXd &u, const ArrayXd &uu, const ArrayXd &v, const ArrayXd &vv, u_int16_t d); // non local non linear function
ArrayXd mlp_call(uint8_t m, uint8_t n, uint8_t l, uint16_t d, ArrayXd &x, double s, double t);
ArrayXd ml_picard(uint8_t m, uint8_t n, uint16_t d, ArrayXd &x, double s, double t, bool start_threads);
void mlp_thread(mlp_t &mlp_args);
void reflect(ArrayXd &a, ArrayXd &b, uint16_t d, double s, double e);

int main() {

	std::string s = eq_name;
	std::cout << s << std::endl << std::endl << std::setprecision(8);

	std::ofstream out_file;
	out_file.open(s + "_mlp.csv");
	out_file << "d, T, n, run, ";
	for (uint8_t i=0; i < rdim; i++) {
		out_file << "result_" << (int)i << ", ";
	}
	out_file << "elapsed_secs" << std::endl;

	double T[1] = {1.0}; //0.1,0.5,
	uint16_t d[1] = {2}; //,2,5,10

	for (uint8_t k = 0; k < sizeof(T) / sizeof(T[0]); k++) {
		for (uint8_t l = 0; l < sizeof(d) / sizeof(d[0]); l++) {
			for (uint16_t j = 0; j < 5; j++) {
				std::chrono::time_point<std::chrono::high_resolution_clock> start_time = std::chrono::high_resolution_clock::now();

				ArrayXd xi = ArrayXd::Zero(d[l], 1);
				ArrayXd result = ml_picard(N_MAX, N_MAX, d[l], xi, 0., T[k], true);

				std::chrono::time_point<std::chrono::high_resolution_clock> end_time = std::chrono::high_resolution_clock::now();
				double elapsed_secs = double(std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count()) / 1000. / 1000.;
				std::cout << "T: " << T[k] << std::endl << "d: " << (int)d[l] << std::endl;
				g(xi);
				std::cout << "g(x): " << tmp << std::endl;
				std::cout << "n: " << (int)N_MAX << std::endl << "Result:" << std::endl << result << std::endl;
				std::cout << "Elapsed secs: " << elapsed_secs << std::endl << std::endl;

				out_file << (int)d[l] << ", " << T[k] << ", " << (int)N_MAX << ", " << (int)j << ", ";
				for (uint8_t i = 0; i < rdim; i++) {
					out_file << result(i) << ", ";
				}
				out_file << elapsed_secs << std::endl;
			}
		}
	}

	out_file.close();

	return 0;
}

ArrayXd f(const ArrayXd &u, const ArrayXd &uu, const ArrayXd &v, const ArrayXd &vv, uint16_t d) {
	fn(u, uu, v, vv, d);
	return ret;
}

void mlp_thread(mlp_t &mlp_args) {
	mlp_args.res = mlp_call(mlp_args.m, mlp_args.n, mlp_args.thread_num, mlp_args.d, mlp_args.x, mlp_args.s, mlp_args.t);
}

ArrayXd mlp_call(uint8_t m, uint8_t n, uint8_t thread_num, uint16_t d, ArrayXd &x, double s, double t) {
	ArrayXd a = ArrayXd::Zero(rdim, 1); // What is returned
	ArrayXd b = ArrayXd::Zero(rdim, 1); // ml_picard from previous step
	ArrayXd b2, b4;

	double r = 0.;
	ArrayXd x2, x3, x32, x34;
	uint32_t num;
	uint32_t loop_num;
	uint32_t remainder;
	static thread_local std::mt19937 generator(128 + clock() + std::hash<std::thread::id>()(std::this_thread::get_id()));
	static thread_local std::normal_distribution<> normal_distribution{0., 1.};
	static thread_local std::uniform_real_distribution<double> uniform_distribution(0., 1.);
	for (uint8_t l = 0; l < std::min(n, (uint8_t)2); l++) {
		b = ArrayXd::Zero(rdim, 1);
		num = (uint32_t)(pow(m, n - l) + 0.5);
		if (num < NUM_THREADS) {
			loop_num = thread_num > num ? 0 : 1;
		} else {
			remainder = m % num;
			if (remainder && thread_num <= remainder) {
				loop_num = num / NUM_THREADS + 1;
			} else {
				loop_num = num / NUM_THREADS;
			}
		}
		for (uint32_t k = 0; k < loop_num; k++) {
			r = s + (t - s) * uniform_distribution(generator);
			x2 = ArrayXd::NullaryExpr(d, [&](){ return normal_distribution(generator); });
			X_sde(s, r, x, x2);
			b2 = ml_picard(m, l, d, x2, r, t, false);
			ArrayXd b3 = ArrayXd::Zero(rdim, 1);
			for (uint32_t h = 0; h < MC_SAMPLES; h++) {
				x3 = ArrayXd::NullaryExpr(d, [&](){ return normal_distribution(generator); });
				x3 = shift_me(x3, x2);
				b3 += f(x2, x3, b2, ml_picard(m, l, d, x3, r, t, false),d);
			}
			b += b3 / ((double) MC_SAMPLES);
		}
		a += (t - s) * (b / ((double)num));
	}
//	std::cout << "thread: " << (int)thread_num << "mid" << std::endl;
	for (uint8_t l = 2; l < n; l++) {
//		if (thread_num == 1) {
//			std::cout << "thread: " << (int)thread_num << " l: " << (int)l << std::endl;
//		}
		b = ArrayXd::Zero(rdim, 1);
		num = (uint32_t)(pow(m, n - l) + 0.5);
		if (num < NUM_THREADS) {
			loop_num = thread_num > num ? 0 : 1;
		} else {
			remainder = m % num;
			if (remainder && thread_num <= remainder) {
				loop_num = num / NUM_THREADS + 1;
			} else {
				loop_num = num / NUM_THREADS;
			}
		}
		for (uint32_t k = 0; k < loop_num; k++) {
			r = s + (t - s) * uniform_distribution(generator);
			x2 = ArrayXd::NullaryExpr(d, [&](){ return normal_distribution(generator); });
			X_sde(s, r, x, x2);
			b2 = ml_picard(m, l, d, x2, r, t, false);
			b4 = ml_picard(m, l - 1, d, x2, r, t, false);
			ArrayXd b3 = ArrayXd::Zero(rdim, 1);
			for (uint32_t h = 0; h < MC_SAMPLES; h++) {
				x3 = ArrayXd::NullaryExpr(d, [&](){ return normal_distribution(generator); });
				x32 = shift_me(x3, x2);
				x34 = shift_me(x3, x2);
				b3 += (f(x2, x32, b2, ml_picard(m, l, d, x32, r, t, false),d) - f(x2, x34, b4, ml_picard(m, l - 1, d, x34, r, t, false),d));
			}
			b += b3 / ((double) MC_SAMPLES);
		}
		a += (t - s) * (b / ((double)num));
	}
//	std::cout << "thread: " << (int)thread_num << "end" << std::endl;
	return a;
}

ArrayXd ml_picard(uint8_t m, uint8_t n, uint16_t d, ArrayXd &x, double s, double t, bool start_threads) {

	if (n == 0) return ArrayXd::Zero(rdim, 1);

	ArrayXd a = ArrayXd::Zero(rdim, 1);
	ArrayXd a2 = ArrayXd::Zero(rdim, 1);
	ArrayXd b = ArrayXd::Zero(rdim, 1);
	ArrayXd b2, b4;


	double r = 0.;
	std::thread threads[NUM_THREADS];
	mlp_t mlp_args[NUM_THREADS];
	ArrayXd x2, x3, x32, x34;
	uint32_t num;
	static thread_local std::mt19937 generator(clock() + std::hash<std::thread::id>()(std::this_thread::get_id()));
	static thread_local std::normal_distribution<> normal_distribution{0., 1.};
	static thread_local std::uniform_real_distribution<double> uniform_distribution(0., 1.);


	if (start_threads) {

		for (uint8_t l = 0; l < NUM_THREADS; l++) {
			mlp_t mlp_arg;
			mlp_arg.m = m;
			mlp_arg.n = n;
			mlp_arg.thread_num = l + 1;
			mlp_arg.d = d;
			mlp_arg.x = x.replicate(1, 1);
			mlp_arg.s = s;
			mlp_arg.t = t;
			mlp_arg.res = 0.;
			mlp_args[l] = mlp_arg;
			threads[l] = std::thread(mlp_thread, std::ref(mlp_args[l]));
		}

		num = (uint32_t)(pow(m, n) + 0.5);
		for (uint32_t k = 0; k < num; k++) {
			x2 = ArrayXd::NullaryExpr(d, [&](){ return normal_distribution(generator); });
			X_sde(s, t, x, x2);
			g(x2);
			a2 += tmp;
		}

		a2 /= (double)num;

		for (uint8_t l = 0; l < NUM_THREADS; l++) {
			threads[l].join();
			a += mlp_args[l].res;
		}

	} else {

		for (uint8_t l = 0; l < std::min(n, (uint8_t)2); l++) {
			b = ArrayXd::Zero(rdim, 1);
			num = (uint32_t)(pow(m, n - l) + 0.5);
			for (uint32_t k = 0; k < num; k++) {
				r = s + (t - s) * uniform_distribution(generator);
				x2 = ArrayXd::NullaryExpr(d, [&](){ return normal_distribution(generator); });
				X_sde(s, r, x, x2);
				b2 = ml_picard(m, l, d, x2, r, t, false);
				ArrayXd b3 = ArrayXd::Zero(rdim, 1);
				for (uint32_t h = 0; h < MC_SAMPLES; h++) {
					x3 = ArrayXd::NullaryExpr(d, [&](){ return normal_distribution(generator); });
					x3 = shift_me(x3, x2);
					b3 += f(x2, x3, b2, ml_picard(m, l, d, x3, r, t, false),d);
				}
				b += b3 / ((double) MC_SAMPLES);
			}
			a += (t - s) * (b / ((double)num));
		}

		for (uint8_t l = 2; l < n; l++) {
			b = ArrayXd::Zero(rdim, 1);
			num = (uint32_t)(pow(m, n - l) + 0.5);
			for (uint32_t k = 0; k < num; k++) {
				r = s + (t - s) * uniform_distribution(generator);
				x2 = ArrayXd::NullaryExpr(d, [&](){ return normal_distribution(generator); });
				X_sde(s, r, x, x2);
				b2 = ml_picard(m, l, d, x2, r, t, false);
				b4 = ml_picard(m, l - 1, d, x2, r, t, false);
				ArrayXd b3 = ArrayXd::Zero(rdim, 1);
				for (uint32_t h = 0; h < MC_SAMPLES; h++) {
					x3 = ArrayXd::NullaryExpr(d, [&](){ return normal_distribution(generator); });
					x32 = shift_me(x3, x2);
					x34 = shift_me(x3, x2);
					b3 += (f(x2, x32, b2, ml_picard(m, l, d, x32, r, t, false),d) - f(x2, x34, b4, ml_picard(m, l - 1, d, x34, r, t, false),d));
				}
				b += b3 / ((double) MC_SAMPLES);
			}
			a += (t - s) * (b / ((double)num));
		}

		num = (uint32_t)(pow(m, n) + 0.5);
		for (uint32_t k = 0; k < num; k++) {
			x2 = ArrayXd::NullaryExpr(d, [&](){ return normal_distribution(generator); });
			X_sde(s, t, x, x2);
			g(x2);
			a2 += tmp;
		}

		a2 /= (double)num;

	}

	return a + a2;
}

void reflect(ArrayXd &a, ArrayXd &b, uint16_t d, double s, double e) {
	double r = 2;
    double rtemp;
    VectorXd n;
    for (uint32_t i = 0; i < d; i++) {
        if (b(i) < s) {
            rtemp = (a(i) - s) / (a(i) - b(i));
            if (rtemp < r) {
                // Crossed left
                r = rtemp;
                n = VectorXd::Zero(d, 1);
                n(i) = - 1;
            }
        } else if (b(i) > e ){
            rtemp = (e - a(i) ) / (b(i) - a(i));
            if (rtemp < r) {
                // Crossed right
                r = rtemp;
                n = VectorXd::Zero(d, 1);
                n(i) = 1;
            }
        }
    }
    while (r < 1) {
        ArrayXd c = a + r * ( b - a );
        a = c;
        b = b - 2 * (n * (b-c).matrix().dot(n)).array();
        r = 2;
        for (uint32_t i = 0; i < d; i++) {
            if (b(i) < s) {
                rtemp = (a(i) - s) / (a(i) - b(i));
                if (rtemp < r) {
                    // Crossed left
                    r = rtemp;
                    n = VectorXd::Zero(d, 1);
                    n(i) = - 1;
                }
            } else if (b(i) > e ){
                rtemp = (e - a(i) ) / (b(i) - a(i));
                if (rtemp < r) {
                    // Crossed right
                    r = rtemp;
                    n = VectorXd::Zero(d, 1);
                    n(i) = 1;
                }
            }
        }
    }
}
