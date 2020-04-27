import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import ipywidgets


def gradient_descent(
        derivative_of_f, *, h_0, alpha, max_iter=10_000, tol=1e-12, verbose=False,
        callback=None
    ):
    """Minimize a (univariate) function f using gradient descent.
    
    Parameters
    ----------
    derivative_of_f : callable
        A function which accepts one argument, h, and outputs the derivative
        of f at h.
    h_0 : float
        The initial guess of the minimizer.
    alpha : float
        The step size parameter.
    max_iter : int
        The maximum number of steps to take.
    tol : float
        The convergence tolerance. If the difference between subsequent guesses
        is less than tol, the algorithm will assume that it has converged.
    verbose : bool
        If `True`, prints the progress of the search.
    callback : callable
        A function called after every update with the new position.
    """
    h = h_0
    for iteration in range(max_iter):
        h_next = h - alpha * derivative_of_f(h)
        if np.linalg.norm(h_next - h) < tol:
            break
        if verbose:
            print(f'iter #{iteration}: h={h_next}')
        if callback is not None:
            callback(h_next)
        h = h_next
    else:
        if verbose:
            print('Reached Max Iters')
    return h


def visualize_gradient_descent(
    *,
    f,
    derivative_of_f,
    h_0,
    alpha,
    n_iters,
    interval_size,
    domain,
    arrow_height=.01
):
    fig, ax = plt.subplots()
    
    # plot the function that we'll be minimizing
    ax.plot(domain, f(domain))

    h = [h_0]
    
    # plot of the current position
    current_ln, = ax.plot(h, [f(h[0])], 'ro', color='black')
    
    # plot of the tangent line
    tangent_ln, = ax.plot([], [], linestyle='--', color='black')
    
    arrows = []
    arrow_height = f(h[0]) * arrow_height

    plt.close()
    
    def init():
        return tangent_ln, current_ln

    def update(frame):
        current_ln.set_data(h[0], [f(h[0])])
        h_next = gradient_descent(
                derivative_of_f,
                h_0=h[0],
                alpha=alpha,
                verbose=False,
                max_iter=1
            )

        arrow = ax.arrow(
            h[0], f(h[0]), (h_next - h[0]), 0, 
            color='black',
            width=.2*arrow_height,
            head_width=arrow_height, 
            head_length=interval_size * .1,
            length_includes_head=True)
        arrows.append(arrow)
        
        for arrow in arrows[:-1]:
            arrow.set_alpha(0.25)

        left_y = f(h[0]) - derivative_of_f(h[0]) * interval_size/2
        right_y = f(h[0]) + derivative_of_f(h[0]) * interval_size/2
        left_x = h[0] - interval_size/2
        right_x = h[0] + interval_size/2
        tangent_ln.set_data([left_x, right_x], [left_y, right_y])

        h[0] = h_next
        
        return tangent_ln, current_ln

    
    ani = animation.FuncAnimation(fig, update, frames=n_iters,
                    init_func=init, blit=True)
    return ani


def plot_tangent_line(
    *,
    f,
    f_prime,
    interval_size,
    domain
):
    def plot(h):
        fig, ax = plt.subplots()
        plt.plot(domain, f(domain))

        left_y = f(h) - f_prime(h) * interval_size/2
        right_y = f(h) + f_prime(h) * interval_size/2
        left_x = h - interval_size/2
        right_x = h + interval_size/2
        plt.plot([left_x, right_x], [left_y, right_y], color='black', linestyle='--')

        plt.scatter(h, f(h), zorder=10)
        
        plt.xlim([25_000, 200_000])
        plt.ylim([np.min(f(domain))*.98, np.max(f(domain)) * 1.02])
        plt.xlabel('Prediction');
        plt.ylabel('MSE');

    return ipywidgets.interact(
        plot, 
        h=ipywidgets.FloatSlider(min=25_000, max=200_000, step=2_000, continuous_update=False)
    )
