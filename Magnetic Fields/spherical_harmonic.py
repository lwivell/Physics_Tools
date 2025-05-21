import json
import numpy as np
import matplotlib.pyplot as plt 
import scipy.special

class SphericalHarmonic(object):
	r"""This class implements a Spherical Harmonic internal field model.
	"""

	def __init__(self, json=None, table=None, nmax=2, table_names=['index','gh','n','m','coefficient','condition']):
		r"""Default constructor.

		:param json: Filename for a JSON file containing the field model.
		:param table: String to a CSV file containing the field model.
		:param nmax: Maximum number of degrees in the model.
		:param table_names: List of strings highlighting the table column names.
		"""
		self._nmax = None
		self._g = None
		self._h = None
		self._name = None
		self._citation = None
		self._shortref = None

		if table is None and json is None:
			self._nmax = nmax
		elif table is None and (not json is None):
			self.load_json(json)
		elif (not table is None) and json is None:
			self.load_table(table, table_names)

	def load_json(self, filename):
		r"""Load a JSON file containing the field model.

		The JSON file should contain (at a minimum) entries nmax, g and h, and
		optionally it can also contain name, citation and shortref. These
		entries store:

		* nmax: maximum degree of the spherical harmonic expansion.
		* g: values of the g quasi-normalised Schmidt coefficients.
		* h: values of the h quasi-normalised Schmidt coefficients.
		* name: convenient name (e.g., JRM09) for the field model.
		* citation: full citation for the model.
		* shortref: shorter citation for the model, e.g., 'Smith et al. (2009)'

		:param filename: Filename for the file we need to load.
		"""
		with open(filename, 'r') as fh:
			data = json.load(fh)

			# make sure we have everything we need
			if 'nmax' not in data:
				raise RuntimeError('Must have the maximum number of degrees in JSON file')

			if 'g' not in data:
				raise RuntimeError('Must have g coefficients in JSON file')

			if 'h' not in data:
				raise RuntimeError('Must have h coefficients in JSON file')

			self._nmax = data['nmax']
			self._g = np.zeros((self._nmax+1,self._nmax+1))
			self._h = np.zeros((self._nmax+1,self._nmax+1))
			for a in data['g']:
				self._g[a['n'],a['m']] = a['value']
			for a in data['h']:
				self._h[a['n'],a['m']] = a['value']
			if 'name' in data:
				self._name = data['name']
			if 'citation' in data:
				self._citation = data['citation']
			if 'shortref' in data:
				self._shortref = data['shortref']

	def save_json(self, filename):
		r"""Save the field model to a JSON file.

		:param filename: Filename for the file we need to save.
		"""
		# Setup the main dictionary.
		data = {}
		data['nmax'] = int(self._nmax)
		data['g'] = []
		data['h'] = []
		if self._citation is not None:
			data['citation'] = self._citation
		if self._shortref is not None:
			data['shortref'] = self._shortref
		if self._name is not None:
			data['name'] = self._name

		# Fill in g and h coefficients.
		for jn in range(1,self._nmax+1):
			for jm in range(0,jn+1):
				data['g'].append({'n': jn, 'm': jm, 'value':self._g[jn,jm]})
				data['h'].append({'n': jn, 'm': jm, 'value':self._h[jn,jm]})

		with open(filename, 'w') as fh:
			json.dump(data, fh)

	def schmidt_leg(self, n, m, x):
		r"""Compute Schmidt quasi-normalised associated Legendre polynomials.

		Calculates the Schmidt quasi-normalised Associated Legendre polynomials
		without a Condon-Shortley phase, using special functions from SciPy.

		The scipy.special.lpmv function returns Associated Legendre polynomials
		that are un-normalised (:math:`P_{n,m}`) and include the Condon-Shortley
		phase :math:`(-1)^m`. For magnetic field modelling we require a Schmidt
		quasi-normalisation where the normalisation is :math:`1` for :math:`m=0`
		and :math:`\sqrt{2\frac{(n-m)!}{(n+m)!}` otherwise. This function
		applies this normalisation and also removes the Condon-Shortley phase.

		References: [Winch+2005]_, [WolframCondon-ShortleyPhase]_

		:param n: Degree of the associated Legendre Polynomial.
		:param m: Order of the associated Legendre Polynomial.
		:param x: Value(s) to evaluate the polynomials at (scalar/array).
		:return: Scalar/array for n and m.
		"""
		if m==0:
			normalisation = 1.0;
		else:
			normalisation = np.sqrt(2.0*np.math.factorial(n-m)/np.math.factorial(n+m))
		return scipy.special.lpmv(m,n,x) * normalisation * ((-1)**(-m))

	def schmidt_dleg(self, n, m, x):
		r"""Derivatives of Schmidt q-normalised Associated Legendre polynomials.

		Calculates derivatives of Schmidt quasi-normalised Associated Legendre
		polynomials without a Condon-Shortley phase, using special functions
		from SciPy and using the recurrence relation:

		.. math::
			(x^2-1)dP_n^m(x) = -(n+1)xP_n^m + (n+1-m)P_{n+1}^m

		We calculate :math:`P_n^m` and :math:`P_{n+1}^m` using SciPy, being
		careful to remove the Condon-Shortley phase before applying this
		identity. Then the polynomials are re-normalised.


		References: [Winch+2005]_, [Westra]_

		:param n: Degree of the associated Legendre Polynomial.
		:param m: Order of the associated Legendre Polynomial.
		:param x: Value(s) to evaluate the polynomials at (scalar/array).
		:return: Scalar/array for n and m.
		"""
		if m==0:
			normalisation = 1.0;
		else:
			normalisation = np.sqrt(2.0*np.math.factorial(n-m)/np.math.factorial(n+m))
		a = scipy.special.lpmv(m,n,x)*((-1)**(-m))
		b = scipy.special.lpmv(m,n+1,x)*((-1)**(-m))

		# Identity: (x^2 - 1)dPnm/dx = -(n+1)xPnm + (n-m+1)P(n+1)m
		dleg = (-(n+1)*x*a + (n-m+1)*b)/(x**2-1)
		return dleg*normalisation

	def field(self, r, th, ph, verbose=False, boundnmin=None, boundnmax=None):
		r"""Calculate the field at a series of points in spherical polar coords.

		:param r: Radial position (same units as the field model).
		:param th: Polar angle (co-latitude) [radians].
		:param ph: Azimuthal angle (positive prograde) [radians].
		:param verbose: Output information on the progress of the routine.
		:param boundnmin: For this call, restrict the lower value of n.
		:param boundnmax: For this call, restrict the upper value of n.
		:return: Tuple (Br,Bth,Bph) in the same units as the field coefficients.

		.. note:: The units for the radial position *must* match the units used
			for the field model. For example, if the radial position is given in
			units of Saturn radius, and one Saturn radius is assumed to be 60268 km,
			but the field model is constructed in units where this is 60330 km,
			then the calculated components will be in error.

		.. todo:: Add assumed radius to the JSON file and to store in the field
			model file.

		References: [Winch+2005]_, [Westra]_
		"""
		Br = np.zeros(r.shape)
		Bth = np.zeros(r.shape)
		Bph = np.zeros(r.shape)
		costh = np.cos(th)
		sinth = np.sin(th)

		if boundnmin is None:
			nmin = 1
		if boundnmax is None:
			nmax = self._nmax
		if (boundnmin is not None) and (boundnmax is not None):
			if boundnmin>boundnmax:
				raise ValueError('For bounds on spherical harmonics, nmin<nmax')
			if boundnmin<1:
				raise ValueError('Minimum bound must be greater than zero')
			nmin=boundnmin

			if boundnmax>self._nmax:
				print('Model Progress: bound greater than maximum, limiting to maximum.')
				nmax = self._nmax
			else:
				nmax = boundnmax

		for jn in range(nmin,nmax+1):
			A = np.power(r,-jn-2)
			for jm in range(0,jn+1):
				if verbose:
					print('Model Progress: n={} m={} g={} h={}'.format(
							jn, jm, self._g[jn,jm], self._h[jn,jm]))

				# Calculate the associated legendre polynomials and their
				# derivatives.
				leg = self.schmidt_leg(jn,jm,costh)
				dleg = self.schmidt_dleg(jn,jm,costh)

				# Calculate the azimuthal term with the gauss coefficients.
				C = self._g[jn,jm]*np.cos(jm*ph) + self._h[jn,jm]*np.sin(jm*ph)

				# Compute field components
				Br = Br + A*(jn+1.0)*leg*C
				Bth = Bth + A*dleg*np.sin(th)*C
				Bph = Bph + A*leg*jm*(self._g[jn,jm]*np.sin(jm*ph)
									-self._h[jn,jm]*np.cos(jm*ph))/sinth

		return Br, Bth, Bph