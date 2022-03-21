import numpy as np
import numba as _numba
from scipy import spatial
from scipy.optimize import minimize

from tqdm.auto import tqdm

class GridAlignment():
    def __init__(self, grid, points, grid_weights=None, point_weights=None, recenter=False):
        self.grid = np.copy(grid)
        self.points = np.copy(points)

        if grid_weights is None:
            self.grid_weights = np.ones(grid.shape[0])
        else:
            self.grid_weights = grid_weights / np.max(grid_weights)
        
        if point_weights is None:
            self.point_weights = np.ones(points.shape[0])
        else:
            self.point_weights = point_weights / np.max(point_weights)
        
        if recenter:
            self.grid[:,0] -= np.mean(self.grid[:,0])
            self.grid[:,1] -= np.mean(self.grid[:,1])
            self.points[:,0] -= np.mean(self.points[:,0])
            self.points[:,1] -= np.mean(self.points[:,1])
        
        self.transformGrid([0,0,1,1,0])
    
    def squareNNDist(self,tm):
        self.transformGrid(tm)
        return np.sum(np.multiply(self.nn[0],self.grid_weights[self.nn[1]],self.point_weights)**2, dtype=np.float64) / self.points.shape[0]
    
    # [dx, dy, dsx, dsy, dt]
    def transformGrid(self,tm):
        self.dx = tm[0]
        self.dy = tm[1]
        self.dsx = tm[2]
        self.dsy = tm[3]
        self.dt = tm[4]

        self.gridTran[:,0]*=self.dsx
        self.gridTran[:,1]*=self.dsy
        self.gridTran[:,0]+=self.dx
        self.gridTran[:,1]+=self.dy
        rotMat = np.array([[np.cos(self.dt),-np.sin(self.dt)],[np.sin(self.dt),np.cos(self.dt)]])
        self.gridTran = np.dot(self.grid,rotMat)
        self.nnTree = spatial.cKDTree(self.gridTran)
        self.nn = self.nnTree.query(self.points)
    
    def align(self, bounds):
        fval = minimize(self.squareNNDist, [self.dx, self.dy, self.dsx, self.dsy, self.dt], bounds=bounds, method="L-BFGS-B")
        self.dx, self.dy, self.dsx, self.dsy, self.dt = fval.x
        return self.squareNNDist([self.dx,self.dy,self.dsx,self.dsy,self.dt])
    
    def roughClock(self, gridsize, steps):
        minObj = self.squareNNDist([0,0,1,1,0])
        minTheta = 0
        minDx = 0
        minDy = 0
        for dx in tqdm(range(-steps,steps + 1)):
            dx *= gridsize
            for dy in range(-steps,steps + 1):
                dy *= gridsize
                for thetaId in range(360):
                    theta = thetaId*np.pi/180
                    obj = self.squareNNDist([dx,dy,self.dsx,self.dsy,theta])
                    if obj<minObj:
                        minObj = obj
                        minTheta=theta
                        minDx = dx
                        minDy = dy
        self.dt = minTheta
        self.dx = minDx
        self.dy = minDy
        return self.squareNNDist([self.dx,self.dy,self.dsx,self.dsy,self.dt])

class LocalizationCluster:
    def __init__(self,grid,localizations,globalDeltaX,recenter):
        """Class container for localization and grid points, fitting functions"""
        #grid NX2 array of x,y localizations NX3 array of x,y,prec
        self.grid = np.copy(grid)
        self.gridTran = np.copy(grid)
        self.localizations = np.copy(localizations)
        self.globalDeltaX = globalDeltaX
        #self.weight = 1/np.sqrt(localizations[:,2]**2 + self.globalDeltaX**2)
        self.nnTree = []
        self.nn = []
        self.dx = 0
        self.dy = 0
        self.dt = 0
        
        
        #Center grid and localizations on 0,0
        xGAve = 0
        yGAve = 0
        gCount = 0
        self.xLAve = 0
        self.yLAve = 0
        lCount = 0
        self.uAve=0

        #just calculating avg x/y/uncertainties and subtracting x/y avg from each coordinate to center the grid

        # START RECENTER
        for row in range(self.grid.shape[0]):
            xGAve+=self.grid[row,0]
            yGAve+=self.grid[row,1]
            gCount+=1
            
        for row in range(self.localizations.shape[0]):
            self.xLAve+=self.localizations[row,0]
            self.yLAve+=self.localizations[row,1]
            self.uAve += self.localizations[row,2]
            lCount+=1
        
        xGAve/=gCount
        yGAve/=gCount
        
        self.xLAve/=lCount
        self.yLAve/=lCount
        self.uAve/=lCount
        self.locCount = lCount
        
        self.xCAve = self.xLAve
        self.yCAve = self.yLAve
        
        if recenter:
            for row in range(self.grid.shape[0]):
                self.grid[row,0]-=xGAve
                self.grid[row,1]-=yGAve
                
            for row in range(self.localizations.shape[0]):
                self.localizations[row,0]-=self.xLAve
                self.localizations[row,1]-=self.yLAve
                
            self.xCAve = 0
            self.yCAve = 0                
        # END RECENTER
            
        self.weightDistMatrix = np.zeros((self.localizations.shape[0],self.gridTran.shape[0]))
        self.wdmComputed = False
        self.objectiveValue = 0
        self.hist = []
        self.signal = []
        self.hist2 = np.ones((self.gridTran.shape[0]+1))
        self.rec = []
        self.kernel=[]
        self.sigma = 0
        self.cHull = spatial.ConvexHull(self.localizations[:,0:2])
        self.area = self.cHull.volume #For 2d points, the internal area is the volume reported by ConvexHull
        
    # applies translation/rotation then calculates nearest neighbors tree for localizations
    def transformGrid(self,tm):
        self.dx = tm[0]
        self.dy = tm[1]
        self.dt = tm[2]
        rotMat = np.array([[np.cos(self.dt),-np.sin(self.dt)],[np.sin(self.dt),np.cos(self.dt)]])
        self.gridTran = np.dot(self.grid,rotMat)
        self.gridTran[:,0]+=self.dx
        self.gridTran[:,1]+=self.dy
        self.nnTree = spatial.cKDTree(self.gridTran)
        self.nn = self.nnTree.query(self.localizations[:,0:2])
        self.wdmComputed = False
    
    # calculates distances of localizations from closest grid points weighted by localization uncertainty and global precision error
    # higher uncertainty = less weighting!!!
    def squareNNDist(self,tm):
        """Weighted mean square nearest neighbor distance, computed by translating grid"""
        self.transformGrid(tm)
        weight = 1/np.sqrt(self.localizations[:,2]**2 + self.globalDeltaX**2)
        self.objectiveValue = float(sum(np.multiply(self.nn[0],weight)**2))/self.locCount
        
        return self.objectiveValue
    
    # calculates distances of localizations from closest grid points
    def squareNNDistUnweighted(self,tm,gridsize):
        """unweighted mean square nearest neighbor distantce, computed by translating grid"""
        self.transformGrid(tm)
        return float(sum((2*self.nn[0]/gridsize)**2))/self.locCount
    
    # calculates squared distances of localizations from (0, 0)
    def rmsDist(self):
        """RMS distance from origin"""
        return self._rmsDistComp(self.localizations)
    
    # compiled version of rmsDist
    @staticmethod
    @_numba.jit(nopython = True)
    def _rmsDistComp(localizations):
        locCount = localizations.shape[0]
        sqDist = 0
        
        for row in range(locCount):
            sqDist += localizations[row,0]**2 + localizations[row,1]**2
            
        sqDist/=locCount
        return sqDist
    
    # rotates grid by 360 degrees until minimum alignment w/ localizations found
    # returns minimum squared nearest-neighbor distance found
    def roughClock(self):
        """Find approximate clocking"""
        minObj = self.squareNNDist([self.dx,self.dy,0])
        minTheta = 0
        for thetaId in range(360):
            theta = thetaId*np.pi/180
            obj = self.squareNNDist([self.dx,self.dy,theta])
            if obj<minObj:
                minObj = obj
                minTheta=theta
        self.dt = minTheta
        self.wdmComputed = False
        return self.squareNNDist([self.dx,self.dy,self.dt])
    
    def computeWeightDistMatrix(self):
        """Compute weighted distance likelihood value, used for likelihood score"""
        self.weightDistMatrix = self._computeWeightDistMatrixComp(self.localizations[:,0:3],self.gridTran,self.localizations.shape[0],self.gridTran.shape[0],self.globalDeltaX,self.area)
        self.wdmComputed=True
    
    @staticmethod
    @_numba.jit(nopython = True)
    def _computeWeightDistMatrixComp(localizations,grid,locCount,gridCount,globalDeltaX,area):
        weightDistMatrix = np.zeros((locCount,gridCount+1))

        # loop over localizations and emitters
        for locId in range(locCount):
            for gridId in range(gridCount):
                # uncertainty squared, gives calculation for normalization constant a = 2pi(delta_x^2 + delta_x_g^2)
                sigma2 = (localizations[locId,2]**2+globalDeltaX**2)

                # likelihood score calculation, does not yet include B / A and P(N, I, B)
                # also divides by 2sigma2 instead of just sigma2?
                weightDistMatrix[locId,gridId] = (1.0/2.0/np.pi/sigma2)*np.exp(-((localizations[locId,0]-grid[gridId,0])**2+(localizations[locId,1]-grid[gridId,1])**2)/(2.0*sigma2))
            weightDistMatrix[locId,gridCount] = 1/area
        return weightDistMatrix

    # calculate negative log likelihood function
    def likelihoodScore(self,image): 
        """log likelihood function for localization positions given intensity values at grid points"""
        if not self.wdmComputed:
            self.computeWeightDistMatrix()
        
        score = self._likelihoodScoreComp(self.weightDistMatrix,image,self.locCount)
        
        return score
    
    
    @staticmethod
    @_numba.jit(nopython = True)
    def _likelihoodScoreComp(weightDistMatrix,image,locCount):
        # calculates first portion of negative log likelihood
        # weightDistMatrix = 2D locs x gridshape + 1
        # image = 1D gridshape + 1
        # image represents intensities at k'th emitter
        score = -(np.sum(np.log(np.dot(weightDistMatrix,image))))
        
        # suppose this and below calculates -ln(B / A * P(N, I, B))?
        # note 'image' is the count of localizations associated with each emitter
        imageSum = np.sum(image)

        # error check - avoid divide by 0
        if imageSum <= 0:
            imageSum = .000001
        score -= imageSum*np.log(locCount/imageSum) + imageSum
        
        return score
    
    # this function is never used even though i just spent like 30 minutes understanding it
    def pixelate(self,gridShape,distThreshold):
        """Produce maximum likelihood pixelated image"""
        # if no grid transforms have taken place, then create cKDTree anyway
        if self.nnTree == []:
            self.nnTree = spatial.cKDTree(self.gridTran)
            self.nn = self.nnTree.query(self.localizations[:,0:2])
        
        # take only localizations within a certain distance from the grid emitters
        self.localizations = self.localizations[self.nn[0]<distThreshold,:]
        self.locCount = self.localizations.shape[0]
        
        # looks like the coder had some sass too
        self.cHull = spatial.ConvexHull(self.localizations[:,0:2])
        self.area = self.cHull.volume #Ugh, scipy
        
        # count how many localizations belong to each emitter (and some extra index?)
        self.hist = np.zeros((self.gridTran.shape[0]+1))
        
        for idx in self.nn[1]:
            self.hist[idx] += 1
      
        bounds = [(0,1e10) for item in self.hist]
        
        # minimization of NLL to find ideal image
        out = minimize(self.likelihoodScore,self.hist,bounds=bounds, method="L-BFGS-B",options={'eps':1e-10})
        self.hist2 = out.x
        #print(self.hist2[-1])
        
        Iscale = max(self.hist)*2
        # normalize hist
        self.signal = (self.hist[:-1].reshape(gridShape))/Iscale

        # reconstruction, # of localizations belonging to each emitter
        self.rec = self.hist2[:-1].reshape(gridShape)

        # renormalize so sum of self.rec SHOULD match original self.hist
        self.rec *= sum(sum(self.signal))/sum(sum(self.rec))*Iscale
        
        return self.likelihoodScore(self.hist2)
    
    # meat of your MLE grid alignment on a single origami
    def fitAndPixelate(self,gridShape,distThreshold,maxIter,toler):
        """Fit grid and produce maximum likelihood pixelated image"""
            
        self.hist = np.zeros((self.gridTran.shape[0]+1))
        
        bounds =[(0,1e10) for item in self.hist]
        
        # find min self.dt angle
        self.roughClock()

        self.hist2 = np.ones((self.gridTran.shape[0]+1))
        lastFVal = self.likelihoodScore(self.hist2)
        #print(lastFVal)
        it=0

        for _ in range(maxIter):
            it+=1

            # minimize transform, then apply it
            out1 = minimize(self.gridLikelihoodScore,[self.dx,self.dy,self.dt],method='Nelder-Mead')
            self.transformGrid(out1.x)
            
            # minimize NLL func
            out2 = minimize(self.likelihoodScore,self.hist2,bounds=bounds, method="L-BFGS-B", options={'eps':1e-10})
            self.hist2 = out2.x
            
            # minimize global localization uncertainty? then apply it
            out3 = minimize(self.gdxLikelihoodScore,[self.globalDeltaX],bounds=[(0,1)],method="L-BFGS-B")
            self.globalDeltaX = out3.x[0]
            FVal = out3.fun
            
            # on third iteration, i guess filter the localizations and recalculate area
            # weird
            # the initial guess may filter out too many localizations, so they only do it after a few iterations of fitting
            if it==3:
                self.localizations = self.localizations[self.nn[0]<distThreshold,:]
                self.locCount = self.localizations.shape[0]
                self.cHull = spatial.ConvexHull(self.localizations[:,0:2])
                self.area = self.cHull.volume #Ugh, scipy
            
            # after 4+ iterations if percent difference between likelihoods is small, break
            if 2.0*np.abs((lastFVal-FVal)/(lastFVal+FVal)) < toler and it > 4:
                break

            lastFVal = FVal
        
        # apply final transform
        self.transformGrid([self.dx,self.dy,self.dt])

        # calculate number of localizations associated with each emitter
        for idx in self.nn[1]:
            self.hist[idx] += 1
        
        # same normalization
        Iscale = max(self.hist)*2     
        self.signal = (self.hist[:-1].reshape(gridShape))/Iscale
        self.rec = self.hist2[:-1].reshape(gridShape)
        self.rec *= sum(sum(self.signal))/sum(sum(self.rec))*Iscale
        
        return out3.fun,it
    
    # calculate likelihood of current intensities with a certain transform
    def gridLikelihoodScore(self,tm):
        """Likelihood score for current image (hist2) given grid coordinate"""
        self.transformGrid(tm)

        # normalization
        image = np.copy(self.hist2)
        image *= self.locCount/sum(image)
        return self.likelihoodScore(image)
    
    # calculate likelihood of current intensities with a certain global delta x
    def gdxLikelihoodScore(self,gdx):
        """Likelihood score for current image (hist2) given global delta x"""
        self.globalDeltaX = gdx[0]
        self.wdmComputed = False

        # normalization
        image = np.copy(self.hist2)
        image *= self.locCount/sum(image)
        return self.likelihoodScore(image)
