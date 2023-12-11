import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt

from matplotlib import cm
from scipy import integrate
from astropy.coordinates import SkyCoord

class EventCalculator():
    _f02 = 1
    _a01 = 1
    _gamma50 = 1
    _newtonG = 6.674e-11 * u.newton * u.m**2 / u.kg**2
    _speedOfLight = 2.98e8 * u.m / u.s
    _internalMotionRMS = _speedOfLight / 2
    _dmHaloA = 1.15e9 * u.solMass / u.kpc**(3/4)
    _mwSkyCoordinates = [8 * u.kpc, SkyCoord("17h45m40", "−29d00m28.17s")]
    _mwMass = 1.15e12 * u.solMass

    @property
    def f02(self):
        return self._f02

    @property
    def a01(self):
        return self._a01

    @property
    def gamma50(self):
        return self._gamma50

    @property
    def newtonG(self):
        return self._newtonG
    @property
    def speedOfLight(self):
        return self._speedOfLight

    @property
    def internalMotionRMS(self):
        return self._internalMotionRMS

    @property
    def dmHaloA(self):
        return self._dmHaloA

    @property
    def mwSkyCoordinates(self):
        return self._mwSkyCoordinates

    @property
    def mwMass(self):
        return self._mwMass

    def __init__(self, configDict):
        self.tensions = configDict.get("tensions", np.logspace(-15, -8, num=8))
        self.curlyG = configDict.get("curlyG", 1e2)
        self.hostGalaxySkyCoordinates = configDict.get("hostGalaxySkyCoordinates",
                                                       [780 * u.kpc,
                                                        SkyCoord("00h42m44.3s", "+41d16m9s")])
        self.hostGalaxyMass = configDict.get("hostGalaxyMass", 2 * 1.15e12 * u.solMass)
        self.sourceSkyCoordinates = configDict.get("sourceSkyCoordinates", None)
        self.results = dict(eventRates = None,
                            dmRho = None,
                            lineOfSight = None,
                            enhancementFactor = None,
                            rStepSize = None)

    @staticmethod
    def _foo(y):
        return -0.337 - (0.064 * y**2)

    @staticmethod
    def skyCoordinatesToCartesian(skyCoordinates):
        d = skyCoordinates[0]
        ra = skyCoordinates[1].ra.radian
        dec = skyCoordinates[1].dec.radian
        cartesianCoords = d * np.array([np.cos(ra) * np.cos(dec),
                                        np.sin(ra) * np.cos(dec),
                                        np.sin(dec)])
        return cartesianCoords

    @staticmethod
    def cartesianToSkyCoordinates(cartesianCoordinates):
        d = np.linalg.norm(cartesianCoordinates)
        dec = np.arcsin(cartesianCoordinates[2] / d)
        ra = np.arctan2(cartesianCoordinates[1], cartesianCoordinates[0])
        return [d, SkyCoord(ra, dec, unit=u.radian)]

    @staticmethod
    def relationalPositionToSkyCoordinates(hostGalaxySkyCoordinates,
                                           sourceDistanceFromHostGalaxy,
                                           positionString,
                                           impactParameter):
        hostGalaxyCenter = EventCalculator.skyCoordinatesToCartesian(hostGalaxySkyCoordinates)
        impactVector = np.cross(hostGalaxyCenter, [1, 0, 0])
        impactVectorNormalized = impactVector / np.linalg.norm(impactVector)

        if positionString == "behind":
            sourcePositionVector = hostGalaxyCenter + impactParameter * impactVectorNormalized
            sourcePositionVectorNormalized = sourcePositionVector / np.linalg.norm(sourcePositionVector)
            sourcePositionVector += np.sqrt(sourceDistanceFromHostGalaxy**2
                                            - impactParameter**2) * sourcePositionVectorNormalized
        elif positionString == "plane":
            sourcePositionVector = hostGalaxyCenter + sourceDistanceFromHostGalaxy * impactVectorNormalized
        elif positionString == "front":
            hostGalaxyCenterNormalized = hostGalaxyCenter / np.linalg.norm(hostGalaxyCenter)
            sourcePositionVector = (hostGalaxyCenter -
                                    sourceDistanceFromHostGalaxy * hostGalaxyCenterNormalized)
        return EventCalculator.cartesianToSkyCoordinates(sourcePositionVector)

    def betaOfMu(self):
        return 10**self._foo(np.log10(self.tensions * 1e15))

    def calculateHaloValues(self, hostGalaxyDistance):
        galaxyMassRatio = self.hostGalaxyMass / self.mwMass
        haloR1 = (1 / (1 + galaxyMassRatio**(1/3))) * hostGalaxyDistance
        haloC = ((hostGalaxyDistance - haloR1) / haloR1)**(9/4)
        return haloR1, haloC

    def _calculateEnhancementFactor(self):
        littleHubble = 0.7
        bigHubble = littleHubble * 100 * u.km / u.s / u.Mpc
        newtonG = 6.674e-11 * u.newton * u.m**2 / u.kg**2
        rhoCritical = 3 * bigHubble**2 / (8 * np.pi * newtonG)
        omegaDM = 1/4
        overdensityDM = self.results["dmRho"] / (omegaDM * rhoCritical)
        betas = self.betaOfMu()
        # Column of betas to take advantage of python broadcasting
        betas = betas.reshape(len(betas), 1)

        # tempF can have values less than 1
        tempF = (betas * overdensityDM).decompose()

        # invoking np.where so the enhancement is never less than homogeneous limit (ie 1)
        self.results["enhancementFactor"] = np.where(tempF > 1, tempF, 1)

    def calculate(self, nSteps=10000):
        mu13 = self.tensions * 1e13
        speedOfLight = 2.98e8 * u.m / u.s
        xIntegral = 4/3
        f02 = self.f02
        a01 = self.a01
        gamma50 = self.gamma50
        lg = 0.0206 * gamma50 * mu13 * u.pc

        self._modelDMRho(nSteps)
        self._calculateEnhancementFactor()

        # Integrate F(r)dr
        enhancementIntegral = integrate.trapz(self.results["enhancementFactor"],
                                              axis=1, dx=self.results["rStepSize"])
        eventRates = (0.2 * lg * speedOfLight * self.curlyG * 1.15e-6 * xIntegral
                     * (f02 * np.sqrt(a01) / (gamma50 * mu13)**(3/2)) *
                     enhancementIntegral * u.kpc**(-3))
        self.results["eventRates"] = eventRates


    def _modelDMRho(self, nSteps):
        hostGalaxyCenter = self.skyCoordinatesToCartesian(self.hostGalaxySkyCoordinates)
        homeGalaxyCenter = self.skyCoordinatesToCartesian(self.mwSkyCoordinates)
        sourcePositionVector = self.skyCoordinatesToCartesian(self.sourceSkyCoordinates)
        hostGalaxyDistance = np.linalg.norm(hostGalaxyCenter - homeGalaxyCenter)

        r1, dmHaloC = self.calculateHaloValues(hostGalaxyDistance)
        dmHaloA = self.dmHaloA

        r = np.array([np.linspace(0 * u.kpc, sourcePositionVector[0], num=nSteps).to(u.kpc),
                      np.linspace(0 * u.kpc, sourcePositionVector[1], num=nSteps).to(u.kpc),
                      np.linspace(0 * u.kpc, sourcePositionVector[2], num=nSteps).to(u.kpc)]) * u.kpc

        self.results["lineOfSight"] = r
        self.results["rStepSize"] = np.linalg.norm(r[:, 1] - r[:, 0])

        # rNorm is the distance from homeGalaxyCenter
        homeGalaxyCenter = homeGalaxyCenter.reshape(3, 1)
        hostGalaxyCenter = hostGalaxyCenter.reshape(3, 1)
        rNorm = np.linalg.norm(r - homeGalaxyCenter, axis=0)

        # rPrime is vector from MW center when within r1,
        # vector from hostGalaxyCenter otherwise
        homeGalaxyDistance = np.linalg.norm(r - homeGalaxyCenter, axis=0)
        hostGalaxyDistance = np.linalg.norm(r - hostGalaxyCenter, axis=0)

        useHomeGalaxyHalo = (dmHaloA / homeGalaxyDistance**(9/4) >
                             dmHaloC * dmHaloA / hostGalaxyDistance**(9/4))

        self.results["dmRho"] = np.where(useHomeGalaxyHalo,
                                         dmHaloA / homeGalaxyDistance**(9/4),
                                         dmHaloC * dmHaloA / hostGalaxyDistance**(9/4))

    def plotEnhancement(self, title=None):
        fig, ax = plt.subplots(1, 1, sharex=True, figsize=(12, 8))
        colors = cm.gist_rainbow(np.linspace(0, 1, num=len(self.tensions)))

        enhancement = self.results["enhancementFactor"]
        x = self.results["lineOfSight"]

        for f, t, c in zip(enhancement, self.tensions, colors):
            ax.semilogy(np.linalg.norm(x, axis=0), f, label=f"{t}", color=c)
            ax.set_ylabel(r"Enhancement Factor $\mathcal{F}(r)$", fontsize=18)

        ax.legend(fontsize=14)

        if title is not None:
            fig.suptitle(title, fontsize=24)
        ax.set_xlabel(f"Distance Along Line of Sight ({x.unit})", fontsize=18)
        ax.tick_params(labelsize=16)
        plt.show(fig)

    def sampleDistances(self, seed=None, nSamples=10000, plot=False):
        cdf = (np.cumsum(self.results["enhancementFactor"], axis=1) /
               self.results["enhancementFactor"].sum(axis=1).reshape(8, 1))

        distanceFromObserver = (self.results["lineOfSight"] -
                                self.results["lineOfSight"][0])
        distanceFromObserver = np.linalg.norm(distanceFromObserver, axis=0)
        distanceFromObserverTiled = np.tile(distanceFromObserver, (8, 1))

        rng = np.random.default_rng(seed=seed)
        samples = rng.random(size=nSamples)

        distanceSamples = np.array([getFirstGoodElementInEachRow(distanceFromObserverTiled, cdf > s)
                                    for s in samples])
        distanceSamples = distanceSamples.transpose() * u.kpc

        if plot:
            fig, ax = plt.subplots(figsize=(12, 10))
            colors = cm.gist_rainbow(np.linspace(0, 1, num=len(self.tensions)))

            for y, t, c in zip(distanceSamples, self.tensions, colors):
                ax.hist(y, bins=np.linspace(distanceFromObserver[0], distanceFromObserver[-1], num=100),
                        histtype="step", label=f"{t}", log=True, color=c)

            ax.legend(fontsize=15, loc="upper left")
            ax.set_xlabel("Distance Along Line of Sight (kpc)", fontsize=18)
            ax.tick_params(labelsize=15)
            ax.grid(visible=True)
            plt.show(fig)

        return distanceSamples

    def computeSourceDistance(self):
        return np.linalg.norm(self.results["lineOfSight"][:, -1])

    def computeMaximumLensingTimes(self, stringTheta=np.pi/4):
        sourceDistance = self.computeSourceDistance()
        deficitAngles = 8 * np.pi * self.tensions
        return (sourceDistance * deficitAngles * np.sin(stringTheta) / (4 * self.internalMotionRMS)).decompose()

    def computeLensingTimeSamples(self, distanceSamples, stringTheta=np.pi/4):
        sourceDistance = self.computeSourceDistance()
        deficitAngles = 8 * np.pi * self.tensions.reshape(8, 1)
        timeSamples = (deficitAngles * np.sin(stringTheta) * distanceSamples
                       * (1 - (distanceSamples / sourceDistance)) / self.internalMotionRMS).decompose()
        return timeSamples

    def computeLensingTimePDF(self, stringTheta=np.pi/4, bins=1000):
        enhancementFactors = self.results["enhancementFactor"]
        distances = np.linalg.norm(self.results["lineOfSight"], axis=0)
        lensingTimes = self.computeLensingTimeSamples(distances, stringTheta=stringTheta).to(u.s)

        if isinstance(bins, int):
            lensingTimePDF = np.zeros((len(enhancementFactors), bins))
            lensingTimeBins = np.zeros((len(enhancementFactors), bins+1))
        else:
            lensingTimePDF = np.zeros((len(enhancementFactors), len(bins) - 1))
            lensingTimeBins = np.zeros((len(enhancementFactors), len(bins)))

        for i in range(len(lensingTimePDF)):
            lensingTimePDF[i], lensingTimeBins[i] = np.histogram(lensingTimes[i],
                                                                 weights=enhancementFactors[i],
                                                                 bins=bins)

        lensingTimeBins *= u.s
        lensingTimePDF /= lensingTimePDF.sum(axis=1).reshape(len(enhancementFactors), 1)

        return lensingTimePDF, lensingTimeBins

    def computeLensingTimeCDF(self, stringTheta=np.pi/4, bins=1000):
        lensingTimePDF, lensingTimeBins = self.computeLensingTimePDF(stringTheta=stringTheta, bins=bins)
        lensingTimeCDF = np.cumsum(lensingTimePDF, axis=1)

        return lensingTimeCDF, lensingTimeBins


class ExperimentExpectationsCalculator():
    def __init__(self, experimentParameters, eventCalculator):
        self.experimentDuration = experimentParameters.get("experimentDuration", 10 * u.yr)
        self.observingHoursPerNight = experimentParameters.get("observingHoursPerNight", 8 * u.hr)
        self.filters = experimentParameters.get("filters", ["u", "g", "r", "i", "z", "y"])
        self.visitsInEachFilter = experimentParameters.get("visitsInEachFilter",
                                                           np.array([45, 67, 183, 191, 167, 168]))
        self.detectionThreshold = experimentParameters.get("detectionThreshold", 0.9)
        self.exposureTime = experimentParameters.get("exposureTime", 30 * u.s)
        self.readoutTime = experimentParameters.get("readoutTime", 2 * u.s)
        self.nullDetectionProbability = experimentParameters.get("nullDetectionProbability", 1e-5)
        self.surveyFootprint = experimentParameters["surveyFootprint"]
        self.eventCalculator = eventCalculator
        self.results = dict(nEventsExpected = None,
                            probeableTensions = None,
                            eventProbabilities = None,
                            efficiencies = None,
                            detectionProbabilities = None,
                            nStarsRequired = None)

    def calculate(self, stringTheta=np.pi/4, bins=1000):
        experimentDuration = self.experimentDuration
        surveyHours = experimentDuration * self.observingHoursPerNight / (24 * u.hr)
        nVisitsAll = self.visitsInEachFilter.sum()
        visitTime = self.exposureTime + self.readoutTime
        delta = self.detectionThreshold

        lensingTimesPDF, lensingTimeBins = self.eventCalculator.computeLensingTimePDF(stringTheta=stringTheta,
                                                                                     bins=bins)
        lensingTimeBinMiddles = (lensingTimeBins[:, 1:] + lensingTimeBins[:, :-1]) / 2

        #(tensions, nLensingTimes)
        observableWindowDuration = experimentDuration + lensingTimeBinMiddles

        #(tensions, nLensingTimes) nEvents of duration t expected
        lam = (self.eventCalculator.results["eventRates"].reshape(8, 1)
               * observableWindowDuration * lensingTimesPDF).decompose()

        # (tensions, nLensingTimes) P(event of duration t overlaps with survey | event of duration t)
        observableEventProbabilities = 1 - np.exp(-lam.sum(axis=1))

        averageOverlapTime = ((lensingTimeBinMiddles * experimentDuration) /
                              (lensingTimeBinMiddles + experimentDuration))
        successfulObservationTime = averageOverlapTime + (1 - 2 * delta) * self.exposureTime
        averageObservableTime = (successfulObservationTime * (surveyHours / experimentDuration).decompose())

        unlensedTimeInYrs = (surveyHours - averageObservableTime).to(u.yr).value
        visitTimeInYrs = visitTime.to(u.yr).value
        surveyHoursInYrs = surveyHours.to(u.yr).value

        failObservationJthExposureProbabilities = np.array([((unlensedTimeInYrs - j * visitTimeInYrs) /
                                                             (surveyHoursInYrs - j * visitTimeInYrs))
                                                            for j in range(nVisitsAll)])
        failObservationJthExposureProbabilities = np.where(failObservationJthExposureProbabilities > 0,
                                                           failObservationJthExposureProbabilities, 0)

        # (tensions, nTimes) P(miss event of duration t | event of duration t)
        failAllObservationsProbabilities = np.prod(failObservationJthExposureProbabilities, axis=0)
        efficiencies = 1 - np.sum(failAllObservationsProbabilities *
                                  lensingTimesPDF, axis=1)

        detectionProbabilities = observableEventProbabilities * efficiencies
        nStarsRequired = np.log(self.nullDetectionProbability) / np.log(1 - detectionProbabilities)

        probeable = (self.eventCalculator.computeMaximumLensingTimes(stringTheta=stringTheta)
                     > 10 * self.exposureTime)

        self.results["nEventsExpected"] = lam.sum(axis=1)[probeable]
        self.results["probeableTensions"] = self.eventCalculator.tensions[probeable]
        self.results["eventProbabilities"] = observableEventProbabilities[probeable]
        self.results["efficiencies"] = efficiencies[probeable]
        self.results["detectionProbabilities"] = detectionProbabilities[probeable]
        self.results["nStarsRequired"] = nStarsRequired[probeable]

    def plotResults(self, title=None, figure=None, show=True, **kwargs):
        if figure is None:
            figure, axs = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
        else:
            axs = figure.axes
            assert len(axs) == 4, f"Figure must have 4 axes but instead has {len(axs)}."

        ys = [self.results["eventProbabilities"],
              self.results["efficiencies"],
              self.results["detectionProbabilities"],
              self.results["nStarsRequired"]]

        titles = ["Event Probabilities", "Efficiency",
                  "Detection Probability",
                  f"Number of Stars Required for P(No Detection) = {self.nullDetectionProbability}"]

        for y, t, ax in zip(ys, titles, axs):
            ax.loglog(self.results["probeableTensions"], y, **kwargs)
            ax.set_title(t, fontsize=20)
            ax.tick_params(labelsize=14)
            ax.grid(visible=True)

        axs[-1].set_xlabel(r"G$\mu/c^2$", fontsize=17)
        axs[0].legend(fontsize=14)

        if title is not None:
            figure.suptitle(title, fontsize=24)

        if show:
            plt.show(figure)

    def plotStarsPerPSF(self, psfFWHM, title=None, figure=None, show=True, **kwargs):
        if figure is None:
            figure, ax = plt.subplots(figsize=(12, 10))
        else:
            ax = figure.axes[0]

        y = (np.sqrt(self.results["nStarsRequired"] / self.surveyFootprint)
             * psfFWHM).decompose()

        x = self.results["probeableTensions"]

        ax.loglog(x, y, **kwargs)

        ax.legend(fontsize=16)
        ax.tick_params(labelsize=14)
        ax.grid(visible=True)
        ax.set_xlabel(r"G$\mu/c^2$", fontsize=17)
        ax.set_ylabel("Stars per PSF FWHM", fontsize=17)

        if title is not None:
            figure.suptitle(title, fontsize=24)

        if show:
            plt.show(figure)

def getFirstGoodElementInEachRow(matrix, mask):
    rows, columns = np.where(mask)
    uniqueRows, uniqueIdxs = np.unique(rows, return_index=True)
    uniqueColumns = columns[uniqueIdxs]
    return matrix[uniqueRows, uniqueColumns]
