import numpy as np
import warnings

class PathPlanning(object):
    def __init__(self, start_point, clockwise=-1, debugging=False):
        """
        :params start_point: numpy.array with coordinates of starting point
        :param clockwise: direction in which we want to find path 
                            (if start_point[1]==0 => direction is -1 !!!)

        """
        self.sorted_blue_cones = []
        self.sorted_yellow_cones = []
        self.start_points = [start_point]
        self.clockwise = clockwise
        # parameters of normal line
        self.k = 0
        self.c = None
        self.k_past = 0
        self.debugging = debugging
        if self.debugging:
            self.ks = []
            self.cs = []
            self.direction_changes = []

    def reset(self, start_point, clockwise=-1):
        self.sorted_blue_cones = []
        self.sorted_yellow_cones = []
        self.start_points = [start_point]
        self.clockwise = clockwise
        # parameters of normal line
        self.k = None
        self.c = None
        self.k_past = 0
        if self.debugging:
            self.ks = []
            self.cs = []
            self.direction_changes = []

    def is_already_added(self, point):
        np.all(np.isin(point, self.sorted_yellow_cones))
        return (np.all(np.isin(point, self.sorted_yellow_cones)) or np.all(np.isin(point, self.sorted_blue_cones)))

    def points_above_normal(self, points):
        return points[np.sign((points[:,1] - self.k * points[:,0] - self.c))==np.sign(self.k)*self.clockwise]

    def find_closest_one(self, points):
        closest_index = np.argmin(np.linalg.norm(points-self.start_points[-1], axis=1))
        closest_cone = points[closest_index]
        return closest_cone

    def calculate_center(self, pointB, pointY):
        return np.array([(pointB[0]-pointY[0])/2 + pointY[0], (pointB[1]-pointY[1])/2 + pointY[1]])
    
    def return_stack(self, object_name):
        if object_name == "yellow cones":
            return np.vstack(self.sorted_yellow_cones)
        elif object_name == "blue cones":
            return np.vstack(self.sorted_blue_cones)
        elif object_name == "centers":
            return np.vstack(self.start_points)           

    def find_line_parameters(self, pointB, pointY, normal=True):
        k = (pointB[1]-pointY[1])/(pointB[0]-pointY[0])
        self.k_past = self.k
        self.k = -1/k if normal else k 
        c = pointB[1] - self.k*pointB[0]
        self.c = c
    
    def check_direction(self):
        if self.k != None and (self.k-self.k_past)==0:# and not(self.auxiliary_variable):
            self.clockwise = - self.clockwise

    def find_next_center(self, pointsB, pointsY, step=None, verbose=True):
        self.find_line_parameters(self.start_points[-1], self.start_points[-2])
        self.check_direction()

        if self.debugging:
            self.ks.append(self.k)
            self.cs.append(self.c)
            self.direction_changes.append(self.clockwise)
           

        B_hat = self.points_above_normal(pointsB)
        Y_hat = self.points_above_normal(pointsY)
        #set_trace()
        b = self.find_closest_one(B_hat)
        y = self.find_closest_one(Y_hat)

        if not self.is_already_added(b):
            self.sorted_blue_cones.append(b)
        if not self.is_already_added(y):
            self.sorted_yellow_cones.append(y)

        s = self.calculate_center(b, y)
        self.start_points.append(s)
        if verbose:
            if step != None:
                print("Step {} done!".format(step+1))
                #print(f"y={-1/self.k*b[0]+self.c},k={self.c}, b={self.c}")
            else:
                print("Step done!")

    def find_path(self, B, Y, n_steps, verbose=True):
        if n_steps < 1:
            raise ValueError("Number of steps must be positive!!")
        """
            B is set of all blue cones
            Y is set of all yellow cones
        """
        # initializing loop
        # step 1)
        b_0 = self.find_closest_one(B)
        y_0 = self.find_closest_one(Y)
        self.sorted_blue_cones.append(b_0)
        self.sorted_yellow_cones.append(y_0)

        s_1 = self.calculate_center(b_0, y_0)
        self.start_points.append(s_1)
        if verbose:
            print("Step 1 done!")

        # step 2)
        if n_steps > 1:
            self.find_line_parameters(b_0, y_0, normal=False) # special case of separate line for 2nd step
            B_hat = self.points_above_normal(B)
            Y_hat = self.points_above_normal(Y)
            #set_trace()
            b_1 = self.find_closest_one(B_hat)
            y_1 = self.find_closest_one(Y_hat)
            self.sorted_blue_cones.append(b_1)
            self.sorted_yellow_cones.append(y_1)

            s_2 = self.calculate_center(b_1, y_1)
            self.start_points.append(s_2)
            if verbose:
                print("Step 2 done!")

        #every other step
        if n_steps >2:
            for step in range(n_steps-2):
                try:
                    self.find_next_center(B, Y, step+2, verbose=verbose)
                    if self.debugging:
                        print(f"k={self.k}, c={self.c}, clockwise={self.clockwise}")
                except ValueError as err:
                    # catching specific error
                    if str(err) == "attempt to get argmin of an empty sequence":
                        warnings.warn("Too many iteration!")
                        break
                    else: 
                        raise ValueError(str(err))