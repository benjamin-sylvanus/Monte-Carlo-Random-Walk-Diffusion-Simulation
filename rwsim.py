from readsim import readsim
import numpy as np
import os
import matplotlib.pyplot as plt


class rwsim:

    def __init__(self,
                 reader=0,
                 swc=0,
                 lut=0,
                 index=0,
                 pairs=0,
                 bounds=0,
                 particle_num=0,
                 step_num=0,
                 step_size=0,
                 perm_prob=0,
                 init_in=0,
                 D0=0,
                 d=0,
                 scale=0,
                 tstep=0):

        self.reader = reader
        self.swc = swc
        self.lut = lut
        self.index = index
        self.pairs = pairs
        self.bounds = bounds
        self.particle_num = particle_num
        self.step_num = step_num
        self.step_size = step_size
        self.perm_prob = perm_prob
        self.init_in = init_in
        self.D0 = D0
        self.d = d
        self.scale = scale
        self.tstep = tstep

    def __repr__(self):
        str = 'Attributes: \n'
        for attribute in self.__dict__.keys():
            t = getattr(self, attribute)

            if isinstance(t, np.float64):
                str = str + ("{}\t{}\n".format(attribute,
                                               getattr(self, attribute))) + (("-"*40)+"\n")*2

            elif isinstance(t, np.ndarray):
                arr = getattr(self, attribute)
                str = str + ("{}\t{}\n".format(attribute,
                                               (arr.shape))) + (("-"*40)+"\n")*2
            elif isinstance(t, np.int64):
                str = str + ("{}\t{}\n".format(attribute,
                                               getattr(self, attribute))) + (("-"*40)+"\n")*2

        return str

    def setReader(self, reader):
        self.reader = reader

    def createReader(self, path):
        self.reader = readsim(path)

    def loadSim(self):
        if isinstance(self.reader, readsim):
            params = self.reader.readall()
            for key in params.keys():
                self.__setattr__(key, params[key])
        else:
            raise Exception("Must initialize a reader or pass a reader\n")

        self.particle_num = np.int64(self.particle_num)
        self.step_num = np.int64(self.step_num)
        print(self)

    def alloc_particles(self):
        particles = np.zeros((np.int64(self.particle_num), 3), dtype=np.double)
        # b = af.Array(particles.ctypes.data, particles.shape, particles.dtype.char);

    def alloc_steps(self):
        pass

    def init_particles(self):
        s = self.swc[0, 4] - self.step_size
        x0 = self.swc[0, 1]
        y0 = self.swc[0, 2]
        z0 = self.swc[0, 3]
        n = np.int64(self.particle_num)
        m = 1
        theta = 2 * np.pi * np.random.rand(n, m)
        v = np.random.rand(n, m)
        phi = np.arccos((2 * v) - 1)
        r = np.power(np.random.rand(n, m), (1/3))
        x = x0 + s * r * np.sin(phi) * np.cos(theta)
        y = y0 + s * r * np.sin(phi) * np.sin(theta)
        z = z0 + s * r * np.cos(phi)
        particles = np.hstack(
            (
                x,
                y,
                z,
                np.ones((self.particle_num, 1), dtype=np.double),
                np.zeros((self.particle_num, 1), dtype=np.double)
            )
        )

        """
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(x, y, z, marker=".")
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.set_title(
            'Random Positions for Particle Init: npar = {}'.format(self.particle_num))
        plt.show()
        """

        return particles

    def getNext(self, n, m):
        step = self.step_size

        theta = 2 * np.pi * np.random.rand(n, m)

        v = np.random.rand(n, m)

        phi = np.arccos((2 * v) - 1)

        x = step * np.sin(phi) * np.cos(theta)

        y = step * np.sin(phi) * np.sin(theta)

        z = step * np.cos(phi)

        """
        x = x0 + step * np.sin(phi) * np.cos(theta)
        y = y0 + step * np.sin(phi) * np.sin(theta)
        z = z0 + step * np.cos(phi)
        """
        randomsteps = np.hstack((x, y, z))
        return randomsteps

    def eventloop(self):
        """
        for each particle:

            create a variable to store results

            create a variable for random steps

            init random steps

            for each step:
                get random vector of step

                set vars for position state flag

                enter control statement:
                    if flag
                        return
                    else
                        set next pos

                        check pos

                        enumerate outcome of next pos

                            if update:
                                set curr -> next;
                            else
                                check rand < perm_prob

                                true:
                                    set curr -> next;
                                    set state -> !state
                                false:
                                    set flag -> true

                set copy vars position state flag -> particle

                return particle

            write results
        """
        self.particles = self.init_particles()

        data = np.zeros((self.particle_num, self.step_num, 3))
        for j in range(self.particle_num):

            particle = self.particles[j, :]

            # initialize variable for coords of particle per step
            rwpath = np.zeros((self.step_num, 3))  # shape (step_num x 3)

            # initialize variable for random direction per step
            randomstep = self.getNext(self.step_num, 1)  # shape (step_num x 3)

            for i in range(self.step_num):

                current_direction = randomstep[i, :]

                position = particle[0:3]

                state = np.bool_(particle[3])

                flag = np.bool_(particle[4])

                if (flag):
                    flag = False

                else:
                    next = self.setNext(position, current_direction)

                    collided = self.checkpos(next, state)

                    if (collided.__eq__("INSIDE")):
                        position = next
                        state = True
                    elif (collided.__eq__("OUTSIDE")):
                        position = next
                        state = False
                    elif (collided.__eq__("CROSSIN")):
                        position = next
                        state = True
                    elif (collided.__eq__("CROSSOUT")):
                        # print("Collision")

                        # random walker was outside or crossed boundary
                        if (np.random.rand() < self.perm_prob):
                            position = next
                            state = False
                            print("Enabled Crossout")
                        else:
                            flag = True

                    particle[0:3] = position
                    particle[3] = np.int64(state)
                    particle[4] = np.int64(flag)

                rwpath[i, :] = particle[0:3]

            data[j, :, :] = rwpath

        return data

    def setNext(self, position, current_vector):
        next = position + current_vector
        return next

    def checkpos(self, next, state):
        """
        extract indicies from lut
        check_connections
        determine state
        """
        indicies = self.preprocesses(next, self.bounds, self.lut)

        if (indicies > -1):
            nextinside = self.check_connections(
                indicies, self.index, self.swc, next, state)

            inside = self.insideLogic(state, nextinside)

        else:
            inside = False

        return inside

    def preprocesses(self, next, bounds, lut):
        """
        convert float to voxel
        idx of values only within bounds
        get those values
        extract indicies from lut (x,y,z)
        """
        # we want the floor of our float values: python is 0 based indexing
        voxes = np.int64(np.floor(next))

        # idx of values in bounds
        # (index based bounds) [0:bounds-1]
        idxmin = voxes > -1
        idxmax = voxes < bounds - 1

        # todo check this is right with testing.
        idx = (idxmax & idxmin).all()
        toextract = voxes[idx]
        indicies = lut[toextract[:, 0], toextract[:, 1], toextract[:, 2]] - 1
        return indicies

    def check_connections(self, indicies, index, swc, next, state):
        nextinside = False

        # extract child parent ids
        children = index[indicies, 0] - 1
        parents = index[indicies, 1] - 1

        # set xyz for inside-outside calculation
        nx0 = next[0]
        ny0 = next[1]
        nz0 = next[2]
        swc1 = swc[children, 1:5]
        swc2 = swc[parents, 1:5]

        for i in range(len(children)):
            p1 = swc1[i, :]
            p2 = swc2[i, :]

            x1 = p1[0]
            y1 = p1[1]
            z1 = p1[2]
            r1 = p1[3]
            x2 = p2[0]
            y2 = p2[1]
            z2 = p2[2]
            r2 = p2[3]

            dist = np.float_power(
                (x2 - x1), 2) + np.float_power((y2-y1), 2) + np.float_power((z2-z1), 2)

            if (r1 > r2):
                b = self.pointbasedswc2v(
                    nx0, ny0, nz0, x1, x2, y1, y2, z1, z2, r1, r2, False, dist)
            else:
                b = self.pointbasedswc2v(
                    nx0, ny0, nz0, x2, x1, y2, y1, z2, z1, r2, r1, False, dist)

            # inside ith connection or already inside
            nextinside = b | nextinside

            # if both positions are inside: break

            if (state and nextinside):
                break

        return nextinside

    def insideLogic(self, state, nextinside):

        # both positions within a connection
        if (state and nextinside):
            inside = "INSIDE"

        # current pos in , next pos out
        elif (state and not (nextinside)):
            inside = "CROSSOUT"

         # current pos out, next pos in
        elif (not (state) and nextinside):
            inside = "CROSSIN"

         # both positions outside
        elif (not (state) and not (nextinside)):
            inside = "OUTSIDE"

        else:
            inside = ""

        return inside

    def pointbasedswc2v(self, x0, y0, z0, x2, x1, y2, y1, z2, z1, r2, r1, emptylogical, dist):

        t = ((x0 - x1) * (x2 - x1) + (y0 - y1) *
             (y2 - y1) + (z0 - z1) * (z2 - z1)) / (dist)

        x = x1 + (x2 - x1) * t
        y = y1 + (y2 - y1) * t
        z = z1 + (z2 - z1) * t

        if dist < np.float_power(r1, 2):
            list1 = False
        else:
            list1 = (x - x1) * (x - x2) + (y - y1) * \
                (y - y2) + (z - z1) * (z - z2) < 0

        if list1:
            dist2 = np.float_power(
                x0 - x, 2) + np.float_power(y0 - y, 2) + np.float_power(z0 - z, 2)

            rd = np.abs(r1 - r2)

            # distance from orthogonal vector to p2
            l = np.sqrt(np.float_power((x - x2), 2) +
                        np.float_power((y - y2), 2) + np.float_power((z - z2), 2))

            # distance from p1 -> p2
            L = np.sqrt(dist)

            c = (rd * l) / L
            r = (c + r2) / np.sqrt(1 - (np.float_power((rd / L), 2)))
            pos1 = dist2 < (np.float_power(r, 2))
            # smaller in one line and less than and equal
            pos = pos1

        else:
            pos2 = ((np.float_power((x0 - x1), 2) + np.float_power((y0 - y1), 2) + np.float_power((z0 - z1), 2)) <= np.float_power(r1, 2)
                    ) | ((np.float_power((x0 - x2), 2) + np.float_power((y0 - y2), 2) + np.float_power((z0 - z2), 2)) <= np.float_power(r2, 2))

            pos = pos2

        return pos

        """
        t = ((x0 - x1) * (x2 - x1) + (y0 - y1) * (y2 - y1) + (z0 - z1) * (z2 - z1)) ./ ...
            (dist);
        x = x1 + (x2 - x1) * t;
        y = y1 + (y2 - y1) * t;
        z = z1 + (z2 - z1) * t;

        if dist < r1^2
            list1 = false;
        else
            list1 = (x - x1) .* (x - x2) + (y - y1) .* (y - y2) + (z - z1) .* (z - z2) < 0;
        end

        if list1
            dist2 = (x0 - x) .^ 2 + (y0 - y) .^ 2 + (z0 - z) .^ 2;

            %     r = r1 + sqrt((x-x1).^2 + (y-y1).^2 + (z-z1).^2) /...
            %         sqrt((x2-x1)^2+(y2-y1)^2+(z2-z1)^2) * (r2-r1);

            %     r = ( c + r2 ) / (sqrt ( 1 - ( |r1-r2 | / l ) )

            %     c = ( |r1 - r2| * l ) / L
            rd = abs(r1 - r2);

            % distance from orthogonal vector to p2
            l = sqrt((x - x2) .^ 2 + (y - y2) .^ 2 + (z - z2) .^ 2);

            % distance from p1 -> p2
            L = sqrt(dist);

            c = (rd * l) ./ L;
            r = (c + r2) ./ sqrt(complex(1 - ((rd / L) .^ 2)));
            pos1 = dist2 < (r .^ 2); % smaller in one line and less than and equal
            pos = pos1;
        else
            pos2 = (((x0 - x1) .^ 2 + (y0 - y1) .^ 2 + (z0 - z1) .^ 2) <= ((r1-eps) ^ 2)) | ...
                (((x0 - x2) .^ 2 + (y0 - y2) .^ 2 + (z0 - z2) .^ 2) <= ((r2-eps) ^ 2));
            pos = pos2;
        end
        """
