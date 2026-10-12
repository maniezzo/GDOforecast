import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from sklearn.cluster import KMeans

def draw_ambiguity(x,y,z, drawPoints=False, drawSegments=False, drawHull=False):
   # x,y,z: coordinate punti
   fig = plt.figure(figsize=(8, 8))
   ax = fig.add_subplot(111, projection='3d')
   L = 3 # default da sovrascrivere
   
   if(drawPoints):
      max_abs = max(
         np.max(np.abs(x)),
         np.max(np.abs(y)),
         np.max(np.abs(z))
      )
      
      L = 1.05 * max_abs  # 5% margin
      
      # Draw points
      ax.scatter(x, y, z, color='blue', s=30)
   
   if(drawSegments):
      # Draw segments from origin to each point
      for xi, yi, zi in zip(x, y, z):
         ax.plot([0, xi], [0, yi], [0, zi],
            color='r',
            lw=1,
            alpha=0.9
         )
   
   # Draw custom axes through the origin
   ax.plot([-L, L], [0, 0], [0, 0], color='black', lw=1)  # X
   ax.plot([0, 0], [-L, L], [0, 0], color='black', lw=2)  # Y
   ax.plot([0, 0], [0, 0], [-L, L], color='black', lw=1)  # Z
   
   # Optional origin point
   ax.scatter([0], [0], [0], color='red', s=40)
   
   # Remove panes
   ax.xaxis.pane.set_visible(False)
   ax.yaxis.pane.set_visible(False)
   ax.zaxis.pane.set_visible(False)
   
   # Remove grid
   ax.grid(False)
   
   # Remove ticks
   ax.set_xticks([])
   ax.set_yticks([])
   ax.set_zticks([])
   
   # Hide the default axis lines/box
   ax.xaxis.line.set_color((1, 1, 1, 0))
   ax.yaxis.line.set_color((1, 1, 1, 0))
   ax.zaxis.line.set_color((1, 1, 1, 0))
   
   ax.set_xlim(-L, L)
   ax.set_ylim(-L, L)
   ax.set_zlim(-L, L)
   ax.set_position([0, 0, 1, 1])
   ax.set_box_aspect([1, 1, 1])
   
   if(drawHull):
      # convex hull
      pts = np.column_stack((x, y, z))
      
      # convex hull
      pts = np.column_stack((x, y, z))
      hull = ConvexHull(pts)
      
      poly = Poly3DCollection(
         [pts[s] for s in hull.simplices],
         facecolor='lightblue',
         edgecolor='blue',
         alpha=0.25,
         linewidths=0.1
      )
      
      ax.add_collection3d(poly)
   plt.show()
   return

def draw_samples(x, y, z, labels, plotBary=False, plotDRO=False):
   """
   labels: array of same length as x,y,z containing values in {0,1,2,3}
   """
   writeLabel = False
   
   fig = plt.figure(figsize=(8, 8))
   ax = fig.add_subplot(111, projection='3d')
   
   pts = np.column_stack((x, y, z))
   
   max_abs = np.max(np.abs(pts))
   L = 1.05 * max_abs
   
   # color map for classes
   colors = {
      0: 'gold',
      1: 'limegreen',
      2: 'dodgerblue',
      3: 'crimson'
   }
   
   # draw points class by class
   for cls in np.unique(labels):
      mask = (labels == cls)
      
      ax.scatter(
         x[mask],
         y[mask],
         z[mask],
         color=colors.get(cls, 'black'),
         s=40,
         label=f'Class {cls}'
      )
   offset = 0.01 * L
   
   if(writeLabel):
      for i, (xi, yi, zi) in enumerate(zip(x, y, z)):
         ax.text( xi + offset, yi + offset, zi + offset, str(i),
                  fontsize=8, color='black')
   # Draw segments from origin to each point
   for xi, yi, zi in zip(x, y, z):
      ax.plot([0, xi], [0, yi], [0, zi],
         color='r',
         lw=0.5,
         alpha=0.9
      )

   if(plotDRO or plotBary):
      # Compute barycenters
      barycenters = []
      
      for cls in np.unique(labels):
         mask = (labels == cls)
         
         bx = np.mean(x[mask])
         by = np.mean(y[mask])
         bz = np.mean(z[mask])
         barycenters.append([bx, by, bz])
         
         # draw barycenter
         ax.scatter(bx, by, bz,
            s=300,
            color=colors[cls],
            edgecolor='black',
            marker='*',
            zorder=100,
            label=f'Barycenter {cls}'
         )
      
   if (plotDRO):
      barycenters = np.array(barycenters)
      if len(barycenters) >= 4:
         bhull = ConvexHull(barycenters)
         
         poly = Poly3DCollection(
            [barycenters[s] for s in bhull.simplices],
            facecolor='orange',
            edgecolor='darkred',
            alpha=0.25,
            linewidths=1.0
         )
         
         ax.add_collection3d(poly)
   
   # axes
   ax.plot([-L, L], [0, 0], [0, 0], color='black', lw=1)
   ax.plot([0, 0], [-L, L], [0, 0], color='black', lw=2)
   ax.plot([0, 0], [0, 0], [-L, L], color='black', lw=1)
   
   ax.scatter([0], [0], [0], color='red', s=40)
   
   ax.xaxis.pane.set_visible(False)
   ax.yaxis.pane.set_visible(False)
   ax.zaxis.pane.set_visible(False)
   
   ax.grid(False)
   
   ax.set_xticks([])
   ax.set_yticks([])
   ax.set_zticks([])
   
   ax.xaxis.line.set_color((1, 1, 1, 0))
   ax.yaxis.line.set_color((1, 1, 1, 0))
   ax.zaxis.line.set_color((1, 1, 1, 0))
   
   ax.set_xlim(-L, L)
   ax.set_ylim(-L, L)
   ax.set_zlim(-L, L)
   ax.set_position([0, 0, 1, 1])
   ax.set_box_aspect([1, 1, 1])
   
   ax.legend()
   
   plt.show()
   return

def generate_points(n=30,a = 6.0, b=4.0, c=5):
   # n: number of points
   # a,b,c:Ellipsoid semi-axes
   
   # Generate points uniformly inside the unit sphere
   np.random.seed(666)
   u = np.random.normal(size=(n, 3))
   u /= np.linalg.norm(u, axis=1)[:, None]  # random directions
   
   r = np.random.rand(n) ** (1 / 3)  # correct radial distribution
   u *= r[:, None]
   
   # Scale to ellipsoid
   x = a * u[:, 0]
   y = b * u[:, 1]
   z = c * u[:, 2]
   
   pts = np.column_stack((x, y, z))
   
   kmeans = KMeans(
      n_clusters=4,
      random_state=995,
      n_init=20
   )
   
   labels = kmeans.fit_predict(pts)
   return x,y,z,labels

if __name__ == '__main__':
   x,y,z,labels = generate_points()
   draw_ambiguity([],[],[])
   draw_ambiguity(x,y,z,drawPoints=True)
   draw_ambiguity(x,y,z,drawPoints=True,drawSegments=True)
   draw_ambiguity(x,y,z,drawPoints=True,drawSegments=True,drawHull=True)
   draw_ambiguity(x,y,z,drawPoints=True)
   draw_samples(x, y, z,labels)
   draw_samples(x, y, z,labels,plotBary=True)
   draw_samples(x, y, z,labels,plotDRO=True)
   pass