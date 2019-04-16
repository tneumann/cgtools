import numpy as np
from cgtools.histograms import soft_histogram, soft_histogram_dd


if __name__ == '__main__':
    import sys
    import pylab as pl

    np.random.seed(2)
    rmin, rmax = 0.1, 1.0
    n_angular_bins = 16
    n_radial_bins = 5
    pts = np.random.normal(loc=0, scale=0.3, size=(4, 2))
    #pts = np.array([(0.2, 0.), (0., 0.2), (0, -0.2), (-0.2, 0)])
    #pts = np.random.uniform(low=-0.7, high=0.7, size=(300,2))
    pts_polar = np.column_stack((np.log10(np.sqrt(pts[:,0]**2 + pts[:,1]**2)),
                                    np.arctan2(pts[:,1], pts[:,0])))
    h = soft_histogram_dd(pts_polar, nbins=(n_radial_bins, n_angular_bins), 
                            range=((np.log10(rmin), np.log10(rmax)), (-np.pi, np.pi)), 
                            wrapping=[False, True])
    #pl.scatter(pts[:,1], pts[:,0], marker='x')
    #pl.imshow(h, extent=(-1, 1, 1, -1))
    pl.subplot(121, polar=True)
    r, theta = np.meshgrid(10 ** np.linspace(np.log10(rmin), np.log10(rmax), n_radial_bins+1), 
                            np.linspace(-np.pi, np.pi, n_angular_bins+1))
    print(r)
    print(h.shape)
    print(h)
    pl.pcolormesh(theta, r, h.T, edgecolors=(1,1,1), lw=0.001, vmin=0, vmax=1)
    pl.scatter(pts_polar[:,1], 10 ** pts_polar[:,0], c='w')

    pl.subplot(122)
    pl.scatter(pts_polar[:,1], pts_polar[:,0], c='w')
    pl.imshow(h, extent=(-np.pi, np.pi, np.log10(rmax), np.log10(rmin)), vmin=0, vmax=1)
    pl.show()


    pts = np.random.normal(loc=0, scale=0.3, size=(8, 2))
    pl.subplot(121)
    pl.title("cgtools.histograms.soft_histogram_dd")
    nbins = 20
    h = soft_histogram_dd(pts, (nbins, nbins), ((-1, 1), (-1, 1)))
    pl.scatter(pts[:,1], pts[:,0], marker='x')
    pl.imshow(h, extent=(-1, 1, 1, -1), vmin=0, vmax=1)
    pl.subplot(122)
    pl.title('numpy.histogramdd')
    h = np.histogramdd(pts, bins=(nbins, nbins), range=((-1, 1), (-1, 1)))[0]
    pl.scatter(pts[:,1], pts[:,0], marker='x')
    pl.imshow(h, extent=(-1, 1, 1, -1))
    pl.show()

