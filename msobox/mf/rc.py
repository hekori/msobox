

oc = OC()
integrator = Integrator()




class IntegeratorAPI(Integrator)

    def forward(self, x0, p, q):

        while True:

            code = self.step()

            if code == 1:
                self.rhs[:,:] = numpy.eye(4)

            if code == 2:
                self.rhs[:,:] = numpy.ones(4)
                self.rhs_d[:,:] = numpy.eye(4)





