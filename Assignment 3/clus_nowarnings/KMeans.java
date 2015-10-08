import java.util.*;

public class KMeans extends ClusteringAlgorithm
{
	// Number of clusters
	private int k;

	// Dimensionality of the vectors
	private int dim;
	
	// Threshold above which the corresponding html is prefetched
	private double prefetchThreshold;
	
	// Array of k clusters, class cluster is used for easy bookkeeping
	private Cluster[] clusters;
	
	// This class represents the clusters, it contains the prototype (the mean of all it's members)
	// and memberlists with the ID's (which are Integer objects) of the datapoints that are member of that cluster.
	// You also want to remember the previous members so you can check if the clusters are stable.
	static class Cluster
	{
		float[] prototype;

		Set<Integer> currentMembers;
		Set<Integer> previousMembers;
		  
		public Cluster(int dim)
		{
			prototype = new float[dim];
			
			currentMembers = new HashSet<Integer>();
			previousMembers = new HashSet<Integer>();
		}
	}
	// These vectors contains the feature vectors you need; the feature vectors are float arrays.
	// Remember that you have to cast them first, since vectors return objects.
	private Vector<float[]> trainData;
	private Vector<float[]> testData;

	// Results of test()
	private double hitrate;
	private double accuracy;
	
	public KMeans(int k, Vector<float[]> trainData, Vector<float[]> testData, int dim)
	{
		this.k = k;
		this.trainData = trainData;
		this.testData = testData; 
		this.dim = dim;
		prefetchThreshold = 0.5;
		
		this.hitrate = 0.0;
		this.accuracy = 0.0;
		
		// Here k new cluster are initialized
		clusters = new Cluster[k];
		for (int ic = 0; ic < k; ic++)
			clusters[ic] = new Cluster(dim);
	}


	public boolean train()
	{
	 	//implement k-means algorithm here:
		// Step 1: Select an initial random partioning with k clusters *check*
		// Step 2: Generate a new partition by assigning each datapoint to its closest cluster center
		// Step 3: recalculate cluster centers
		// Step 4: repeat until clustermembership stabilizes
		
		System.out.println("RandomPartition");
		this.randomPartition(); ///step 1
		
		while(!this.sameProtypes()){ /// step 4
			System.out.println("Partition");
			this.partition(); ///step 2
			
			System.out.println("CalculatingPrototypes");
			this.calculateProtoypes(); ///step 3
			
		}
		
		return false;
	}

	private boolean sameProtypes() {
		///Check if the current and previous members of all the clusters are the same.
		
		///For each of the clusters
		for(int cluster = 0; cluster < this.k; cluster++){	
			
			Set<Integer> current = this.clusters[cluster].currentMembers;
			Set<Integer> previous = this.clusters[cluster].previousMembers;
			
			///Check if the two sets are equal
			if(!current.equals(previous)){
				return false;
			}
			
		}
		
		return true;
	}

	
	private void partition() {
		///Repartition the data over the clusters according to Euclidian distance
		
		this.newGeneration();
		
		Iterator<float[]> users = trainData.iterator();
		
		///For every user
		while(users.hasNext()){
			float[] currentUser = users.next();
			double minDistance = Double.MAX_VALUE;
			int chosenCluster = 0;
			
			///Check the distance for every combination with a cluster
			for(int cluster = 0; cluster < this.k; cluster++){
				Cluster currentCluster = this.clusters[cluster];
				
				double distance = this.euclidianDist(currentUser, currentCluster.prototype);
				
				if(distance < minDistance){
					minDistance = distance;
					chosenCluster = cluster;
				}
			}
			
			///Now add the user to the best cluster
			this.clusters[chosenCluster].currentMembers.add(trainData.indexOf(currentUser));
			
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


	private void newGeneration() {
		///Lets the current members become the previous members
		///and creates a new set of current members.
		
		for(int i = 0; i < this.k; i++){
			this.clusters[i].previousMembers = this.clusters[i].currentMembers;
			this.clusters[i].currentMembers = new HashSet<Integer>();
		}
		
	}


	private void randomPartition() {
		/// Select a random partitioning of the clusters
		
		Iterator<float[]> user = trainData.iterator();
		Random randomizer = new Random();
		
		while(user.hasNext()){
			float[] urls = user.next();
			
			///Select to which cluster this member will be assigned
			
			int selectedCluster = randomizer.nextInt(k);
			this.clusters[selectedCluster].currentMembers.add(trainData.indexOf(urls));
			
		}
		
		this.calculateProtoypes();
		
	}


	private void calculateProtoypes() {
		/// Calculate the prototypes of each of the clusters
		
		///Loop over clusters
		for(int i = 0; i < this.k; i++){
			Cluster cluster = this.clusters[i];
			
			///Loop over urls
			for(int url = 0; url < this.dim; url++){
				cluster.prototype[url] = 0.0f;
				Iterator<Integer> user = cluster.currentMembers.iterator();
				
				///Loop over members
				while(user.hasNext()){
					int x = user.next();
					float[] currentUser = trainData.get(x);
					cluster.prototype[url] += currentUser[url];
					
				}
				
				cluster.prototype[url] /= cluster.currentMembers.size();
			}
		}
	}


	public boolean test()
	{	
		int prefetched = 0;
		int requests = 0;
		int hits = 0;
		
		// iterate along all clients. Assumption: the same clients are in the same order as in the testData
		for(int clients = 0; clients < testData.size(); clients++){
			
			// get the actual testData (the vector) of this client
			float[] currentClient = testData.elementAt(clients);
			
			// for each client find the cluster of which it is a member
			for(int cluster = 0; cluster < this.k; cluster++){
				Cluster currentCluster = this.clusters[cluster];
				if(currentCluster.currentMembers.contains(clients)){
					
					// iterate along all dimensions
					for(int url = 0; url < this.dim; url++){
						
						// and count prefetched htmls
						prefetched = currentCluster.prototype[url] >= this.prefetchThreshold ? prefetched+1 : prefetched ; 
						
						// count number of hits
						hits = (currentCluster.prototype[url] >= this.prefetchThreshold) && (currentClient[url] == 1.0) ? hits+1 : hits ;

						// count number of requests
						requests = currentClient[url] == 1.0 ? requests+1 : requests ;
						
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


	// The following members are called by RunClustering, in order to present information to the user
	public void showTest()
	{
		System.out.println("Prefetch threshold=" + this.prefetchThreshold);
		System.out.println("Hitrate: " + this.hitrate);
		System.out.println("Accuracy: " + this.accuracy);
		System.out.println("Hitrate+Accuracy=" + (this.hitrate + this.accuracy));
	}
	
	public void showMembers()
	{
		for (int i = 0; i < k; i++)
			System.out.println("\nMembers cluster["+i+"] :" + clusters[i].currentMembers);
	}
	
	public void showPrototypes()
	{
		for (int ic = 0; ic < k; ic++) {
			System.out.print("\nPrototype cluster["+ic+"] :");
			
			for (int ip = 0; ip < dim; ip++)
				System.out.print(clusters[ic].prototype[ip] + " ");
			
			System.out.println();
		 }
	}

	// With this function you can set the prefetch threshold.
	public void setPrefetchThreshold(double prefetchThreshold)
	{
		this.prefetchThreshold = prefetchThreshold;
	}
}
