import numpy as np
from scipy.fft import fft2,ifft2,fftfreq,fftshift

def wavenumbers(coordinates):
    '''
    Calculates the wavenumber associated with a regular grid.

    input
    shape: tuple of ints 
        Tuple containing the number of point along x- and y-axis
    
    dx,dx : floats
        Grid spacing along x- and y-axis

    output
    kx,ky,kz : arrays 
        A set of arrays with the wavenumber associated with the three axes.

    '''
    y = coordinates[0]
    x = coordinates[1]
    shape = coordinates[0].shape

    nx = shape[0]
    ny = shape[1]

    dx = (x.max() - x.min())/(nx - 1)
    dy = (y.max() - y.min())/(ny - 1)

    ky, kx = np.meshgrid(2*np.pi*fftfreq(ny, dy), 2*np.pi*fftfreq(nx, dx))
    kz = np.sqrt(kx*kx + ky*ky)

    return kx,ky,kz

def transforms(data,filter_fft):
    """
    Compute the convolution between the filter and the transformed data
    """
    data_fft = fft2(data)
    convolve = filter_fft*data_fft
    out = ifft2(convolve).real

    return out

def upward_continuation(coordinates,data,height):
    """
    Calculating the upward/downward continuation
    """
    kx,ky,kz = wavenumbers(coordinates)
    filter_fft = np.exp(-height*kz)
    result = transforms(data,filter_fft)
    return result

def reduction_to_pole(coordinates,data,magnetization,main_field):
    """
    Calculating the reduction to the pole field
    """
    I,D = magnetization
    I0,D0 = main_field

    mx = np.cos(np.deg2rad(I))*np.cos(np.deg2rad(D))
    my = np.cos(np.deg2rad(I))*np.sin(np.deg2rad(D))
    mz = np.sin(np.deg2rad(I))

    j0x = np.cos(np.deg2rad(I0))*np.cos(np.deg2rad(D0))
    j0y = np.cos(np.deg2rad(I0))*np.sin(np.deg2rad(D0))
    j0z = np.sin(np.deg2rad(I0))

    kx,ky,kz = wavenumbers(coordinates)

    a1 = mz*j0z - mx*j0x
    a2 = mz*j0z - my*j0y
    a3 = -my*j0x - mx*j0y
    b1 = mx*j0z + mz*j0x
    b2 = my*j0z + mz*j0y
    kz_sqr = kz*kz
    kx_sqr = kx*kx
    ky_sqr = ky*ky

    filter_fft = kz_sqr/((a1*kx_sqr + a2*ky_sqr + a3*kx*ky + 1e-9) + 1j*kz*(b1*kx + b2*ky + 1e-9))
    result = transforms(data,filter_fft)
    return result

def bx(coordinates,data,main_field):
    """
    Calculating the x-component of the magnetic field from total-field anomaly
    """
    I0,D0 = main_field
    j0x = np.cos(np.deg2rad(I0))*np.cos(np.deg2rad(D0))
    j0y = np.cos(np.deg2rad(I0))*np.sin(np.deg2rad(D0))
    j0z = np.sin(np.deg2rad(I0))
    kx,ky,kz = wavenumbers(coordinates)

    q = j0z*kz + 1j*(j0x*kx + j0y*ky)

    filter_fft = (1j*kx)/(q + 1e-20)
    result = transforms(data,filter_fft)
    return result

def by(coordinates,data,main_field):
    """
    Calculating the y-component of the magnetic field from total-field anomaly
    """
    I0,D0 = main_field
    j0x = np.cos(np.deg2rad(I0))*np.cos(np.deg2rad(D0))
    j0y = np.cos(np.deg2rad(I0))*np.sin(np.deg2rad(D0))
    j0z = np.sin(np.deg2rad(I0))
    kx,ky,kz = wavenumbers(coordinates)

    q = j0z*kz + 1j*(j0x*kx + j0y*ky)

    filter_fft = (1j*ky)/(q + 1e-20)
    result = transforms(data,filter_fft)
    return result

def bz(coordinates,data,main_field):
    """
    Calculating the z-component of the magnetic field from total-field anomaly
    """
    I0,D0 = main_field
    j0x = np.cos(np.deg2rad(I0))*np.cos(np.deg2rad(D0))
    j0y = np.cos(np.deg2rad(I0))*np.sin(np.deg2rad(D0))
    j0z = np.sin(np.deg2rad(I0))
    kx,ky,kz = wavenumbers(coordinates)

    q = j0z*kz + 1j*(j0x*kx + j0y*ky)

    filter_fft = ((kz)/(q + 1e-20))
    result = transforms(data,filter_fft)
    return result

def derivative_x(coordinates,data,order):
    """
    Calculating the derivative of a given order in relation to x
    """
    kx,_,_ = wavenumbers(coordinates)
    n = order
    filter_fft = (1j*kx)**n
    result = transforms(data,filter_fft)
    return result

def derivative_y(coordinates,data,order):
    """
    Calculating the derivative of a given order in relation to y
    """
    _,ky,_ = wavenumbers(coordinates)
    n = order
    filter_fft = (1j*ky)**n
    result = transforms(data,filter_fft)
    return result

def derivative_z(coordinates,data,order):
    """
    Calculating the derivative of a given order in relation to z
    """
    _,_,kz = wavenumbers(coordinates)
    n = order
    filter_fft = (kz)**n
    result = transforms(data,filter_fft)
    return result
