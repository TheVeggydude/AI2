import java.util.ArrayList;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Random;
import java.util.Set;
import java.util.Vector;

public class Kohonen extends ClusteringAlgorithm
{
	// Size of clustersmap
	private int n;

	// Number of epochs
	private int epochs;
	
	// Dimensionality of the vectors
	private int dim;
	
	// Threshold above which the corresponding html is prefetched
	private double prefetchThreshold;

	private double initialLearningRate; 
	
	// This class represents the clusters, it contains the prototype (the mean of all it's members)
	// and a memberlist with the ID's (Integer objects) of the datapoints that are member of that cluster.  
	private Cluster[][] clusters;

	// Vector which contains the train/test data
	private Vector<float[]> trainData;
	private Vector<float[]> testData;
	
	// Results of test()
	private double hitrate;
	private double accuracy;
	
	static class Cluster
	{
			float[] prototype;

			Set<Integer> currentMembers;

			public Cluster()
			{
				currentMembers = new HashSet<Integer>();
			}
	}
	
	private class Coordinate{
		int x;
		int y;
		
		public Coordinate(int x, int y){
			this.x = x;
			this.y = y;
		}
	}
	
	public Kohonen(int n, int epochs, Vector<float[]> trainData, Vector<float[]> testData, int dim)
	{
		this.n = n;
		this.epochs = epochs;
		prefetchThreshold = 0.5;
		initialLearningRate = 0.8;
		this.trainData = trainData;
		this.testData = testData; 
		this.dim = dim;       
		
		Random rnd = new Random();

		//Step 1: Here n*n new cluster are initialized
		clusters = new Cluster[n][n];
		for (int i = 0; i < n; i++)  {
			for (int i2 = 0; i2 < n; i2++) {
				clusters[i][i2] = new Cluster();
				clusters[i][i2].prototype = new float[dim];
				for (int idx = 0; idx < dim; ++idx){
					clusters[i][i2].prototype[idx] = rnd.nextFloat(); ///Step 1
				}
			}
		}
	}
	
	private double euclidianDist(float[] currentUser, float[] prototype) {
		///Calculate the Euclidian distance between the member's array and the prototype
		
		double result = 0;
		
		for(int i= 0; i < this.dim; i++){
			result += Math.pow((currentUser[i] - prototype[i]), 2);
		}
		
		result = Math.sqrt(result);
		
		return result;
	}
	
	private Coordinate findBMU(float[] us){
		///Step 3: find the cluster closest to the input vector (us) in terms of euclidian distance
		double min = Double.MAX_VALUE;
		Coordinate best = new Coordinate(0, 0);
		for (int i1 = 0; i1 < n; ++i1){
			for (int i2 = 0; i2 < n; ++i2){ ///Loop over all clusters
				double dist = euclidianDist(us, clusters[i1][i2].prototype);
				if (dist < min){ ///Select closest
					min = dist;
					best = new Coordinate(i1, i2);
				}
			}
		}
		return best;
	}
	
	private ArrayList<Cluster> findNeighbors(Coordinate c, double r){
		///Step 4: find all clusters (output) in the neighborhood (r) of the BMU (c)
		ArrayList<Cluster> al = new ArrayList<Cluster>();
		for (int i1 = 0; i1 < n; ++i1){
			for (int i2 = 0; i2 < n; ++i2){ ///Loop over all clusters
				if (Math.abs(c.x - i1) <= r && Math.abs(c.y - i2) <= r){
					al.add(clusters[i1][i2]);
				}
			}
		}
		
		return al;
	}
	
	private void updateNeighbors(ArrayList<Cluster> nb, float[] inpVec, float eta){
		//Step 5: Update all neighbors (nb) to be more like the input vector (inpVec)
        ///Awaiting TA reply on whether this is correct method!
		Iterator<Cluster> clusters = nb.iterator();
		
		while (clusters.hasNext()){
			Cluster c = clusters.next();
			for (int i = 0; i < dim; ++i){
				c.prototype[i] = (1 - eta)*(c.prototype[i]+(eta*inpVec[i]));
			}
		}
	}
	
	public boolean train()
	{
		// Step 1: initialize map with random vectors (A good place to do this, is in the initialisation of the clusters) -- DONE
		// Repeat 'epochs' times:
			// Step 2: Calculate the squareSize and the learningRate, these decrease linearly with the number of epochs.
			// Step 3: Every input vector is presented to the map (always in the same order)
			// For each vector its Best Matching Unit is found, and :
				// Step 4: All nodes within the neighbourhood of the BMU are changed, you don't have to use distance relative learning.
		// Since training kohonen maps can take quite a while, presenting the user with a progress bar would be nice
		
		Iterator<float[]> users;
		
		for (int e =0; e < epochs; ++e){///Step 2 and 6
			
			float r = (n/2)*(1-((float) e/epochs)); ///Calculate r every loop as e changes, within the loop would be inefficient
			float eta = 0.8f*(1-((float) e/epochs)); ///Same for eta.
			
			users = trainData.iterator();//(re-)initialize the iterator
			while (users.hasNext()){
				float[] us = users.next();
				Coordinate BMU = findBMU(us);///Step 3
				ArrayList<Cluster> neighbors = findNeighbors(BMU, r);///Step 4
				updateNeighbors(neighbors, us, eta);///Step 5
			}
			
			System.out.println("Epoch: " + e + " | r = " + r + " | eta = " + eta);
		}
		
		///Now that prototypes are trained, add each user to a cluster.
		users = trainData.iterator();
		while (users.hasNext()){
			float[] us = users.next();
			Coordinate BMU = findBMU(us);
			clusters[BMU.x][BMU.y].currentMembers.add(trainData.indexOf(us));
		}
		
		return true;
	}
	
	public boolean test()
	{
		// iterate along all clients
		// for each client find the cluster of which it is a member
		// get the actual testData (the vector) of this client
		
		Iterator<float[]> clients = testData.iterator();
		int prefetched = 0;
		int hits = 0;
		int requests = 0;
		
		
		while(clients.hasNext()){
			
			float[] currentClient = clients.next();
			
			Coordinate c = findBMU(currentClient);
			float[] prototype = this.clusters[c.x][c.y].prototype;
			
			// iterate along all dimensions
			for (int x = 0; x < n; ++x){
				for (int y = 0; y < n; ++y){
					if (clusters[x][y].currentMembers.contains(testData.indexOf(currentClient))){
						///Found matching cluster.
						for(int url = 0; url < this.dim; url++){
							
							// and count prefetched htmls
							prefetched = prototype[url] >= this.prefetchThreshold ? prefetched+1 : prefetched ; 
							
							// count number of hits
							hits = (prototype[url] >= this.prefetchThreshold) && (currentClient[url] == 1.0) ? hits+1 : hits ;

							// count number of requests
							requests = currentClient[url] == 1.0 ? requests+1 : requests ;
							
						}
					}
				}
			}
			
		}
		

		// set the global variables hitrate and accuracy to their appropriate value
		
		System.out.println("Hits: " + hits + ", requests: " + requests +", prefetched: " + prefetched);
		this.hitrate = hits/(double)requests;
		this.accuracy = hits/(double)prefetched;
				
		return true;
	}


	public void showTest()
	{
		System.out.println("Initial learning Rate=" + initialLearningRate);
		System.out.println("Prefetch threshold=" + prefetchThreshold);
		System.out.println("Hitrate: " + hitrate);
		System.out.println("Accuracy: " + accuracy);
		System.out.println("Hitrate+Accuracy=" + (hitrate + accuracy));
	}
 
 
	public void showMembers()
	{
		for (int i = 0; i < n; i++)
			for (int i2 = 0; i2 < n; i2++)
				System.out.println("\nMembers cluster["+i+"]["+i2+"] :" + clusters[i][i2].currentMembers);
	}

	public void showPrototypes()
	{
		for (int i = 0; i < n; i++) {
			for (int i2 = 0; i2 < n; i2++) {
				System.out.print("\nPrototype cluster["+i+"]["+i2+"] :");
				
				for (int i3 = 0; i3 < dim; i3++)
					System.out.print(" " + clusters[i][i2].prototype[i3]);
				
				System.out.println();
			}
		}
	}

	public void setPrefetchThreshold(double prefetchThreshold)
	{
		this.prefetchThreshold = prefetchThreshold;
	}
}

