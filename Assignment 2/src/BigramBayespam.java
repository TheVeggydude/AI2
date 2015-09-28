import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.text.DecimalFormat;
import java.util.Enumeration;
import java.util.Hashtable;
import java.util.StringTokenizer;

public class BigramBayespam {
	// This defines the two types of messages we have.
    static enum MessageType
    {
        NORMAL, SPAM
    }

    // This a class with two counters (for regular and for spam)
    static class Multiple_Counter
    {
        int counter_spam    = 0;
        int counter_regular = 0;

        // Increase one of the counters by one
        public void incrementCounter(MessageType type)
        {
            if ( type == MessageType.NORMAL ){
                ++counter_regular;
            } else {
                ++counter_spam;
            }
        }
    }

    // Listings of the two subdirectories (regular/ and spam/)
    private static File[] listing_regular = new File[0];
    private static File[] listing_spam = new File[0];

    // A hash table for the vocabulary (word searching is very fast in a hash table)
    private static Hashtable <String, Multiple_Counter> vocab = new Hashtable <String, Multiple_Counter> ();

    ///Counters for the amount of a certain type of message (set in readMessages())
    private static int spamCnt;
    private static int normalCnt;
    
    ///Counters for the amount of words in types of messages (set in addWord())
    private static int spamWordCnt;
    private static int normalWordCnt;
    
    ///The tweaker constant declared in section 2.2 is here for tweaking

    private static final double EPSILON = 1;
    
    ///The two parameters suggested by the assignment that cut off bigrams with too short length and too few occurences.
    private static final int MIN_BIGRAM_LENGTH = 9;   ///NOTE: includes a space!
    private static final int MIN_BIGRAM_OCCURANCE = 2;  ///Any bigram occurring (across normal AND spam!) less will get pruned. 
    
    // Add a word to the vocabulary
    private static void addWord(String word, MessageType type)
    {
        Multiple_Counter counter = new Multiple_Counter();

        if ( vocab.containsKey(word) ){                  // if word exists already in the vocabulary..
            counter = vocab.get(word);                  // get the counter from the hashtable
        }
        counter.incrementCounter(type);                 // increase the counter appropriately

        vocab.put(word, counter);                       // put the word with its counter into the hashtable
        
        if (type == MessageType.NORMAL){
            ++normalWordCnt;
        } else {
            ++spamWordCnt;
        }
    }


    // List the regular and spam messages
    private static void listDirs(File dir_location)
    {
        // List all files in the directory passed
        File[] dir_listing = dir_location.listFiles();

        // Check that there are 2 subdirectories
        if ( dir_listing.length != 2 )
        {
            System.out.println( "- Error: specified directory does not contain two subdirectories.\n" );
            Runtime.getRuntime().exit(0);
        }

        listing_regular = dir_listing[0].listFiles();
        listing_spam    = dir_listing[1].listFiles();
    }

    
    // Print the current content of the vocabulary
    private static void printVocab()
    {
        Multiple_Counter counter = new Multiple_Counter();

        for (Enumeration<String> e = vocab.keys() ; e.hasMoreElements() ;)
        {   
            String word;
            
            word = e.nextElement();
            counter  = vocab.get(word);
            
            ///System.out.println( word + " | in regular: " + counter.counter_regular + 
            ///                    " in spam: "    + counter.counter_spam);
            
            System.out.println( word + " | in regular: " + counter.counter_regular + 
            		" (" + getNormalLikelihood(word) + 
            		") in spam: " + counter.counter_spam +
            		" (" + getSpamLikelihood(word) + ")");
        }
    }
    
    private static String toAlpha(String s)
    ///Filters out all non-letters and lowercases it.
    {
    	StringBuffer alpha = new StringBuffer();
    	int idx;
    	for (idx = 0; idx < s.length(); ++idx)
    	{
    		char c = s.charAt(idx);
    		if (Character.isLetter(c))
    		{
    			alpha.append(Character.toLowerCase(c));
    		}
    	}
    	return new String(alpha);
    }


    // Read the words from messages and add them to your vocabulary. The boolean type determines whether the messages are regular or not  
    private static void readMessages(MessageType type)
    throws IOException
    {
        File[] messages = new File[0];

        if (type == MessageType.NORMAL){
            messages = listing_regular;
        } else {
            messages = listing_spam;
        }
        
        for (int i = 0; i < messages.length; ++i)
        {
            FileInputStream i_s = new FileInputStream( messages[i] );
            BufferedReader in = new BufferedReader(new InputStreamReader(i_s));
            String line;
            String old = ""; /// old represents the previous token, is initialised to the first
            
            while ((line = in.readLine()) != null)                      // read a line
            {
                StringTokenizer st = new StringTokenizer(line);         // parse it into words

                while (st.hasMoreTokens())                  // while there are still words left..
                {
                	if (old == "")
                		old = toAlpha(st.nextToken());		///make sure to grab the first token (in all lines) as a different case!
                	
					String next = toAlpha(st.nextToken());
					if (old.length()+next.length() >= MIN_BIGRAM_LENGTH-1)	/// and both words have 4 or more characters
					{														/// The -1 corrects for the space.
						addWord(old + " " + next, type);        // add them to the vocabulary
					}										/// converted and all
					old = next;
				}
            }

            in.close();
            if (type == MessageType.NORMAL){		/// increment the number of messages
                ++normalCnt;
            } else {
                ++spamCnt;
            }
        }
    }
    
/// ------------ Section 2 ------------------------
/// ------------ 2.1 -------------------------
    
    private static double getNormalCount()		/// give number of count messages
    {
    	return normalCnt;
    }
    
    private static double getSpamCount()		/// give number of spam messages
    {
    	return spamCnt;
    }
    
    private static double getTotalCount()		/// give total number of messages
    {
    	return normalCnt;
    }
    
    private static double getPriorNormal()
    {
    	return Math.log(getNormalCount()/getTotalCount()) ;
    }
    
    private static double getPriorSpam()
    {
    	return Math.log(getSpamCount()/getTotalCount());
    }
    
/// -------------- 2.2 ------------------
    
    private static double getNormalWordCount()
    {
    	return normalWordCnt;
    }
    
    private static double getSpamWordCount()
    {
    	return spamWordCnt;
    }
    
    private static double getNormalLikelihood(String word)
    { /// given a word compute its class conditional likelihood P(wj | regular)
    	Multiple_Counter counter  = vocab.get(word);
    	return Math.log(zeroSafeguard(counter.counter_regular / getNormalWordCount()));
    }
    
    private static double getSpamLikelihood(String word)
    {
    	Multiple_Counter counter  = vocab.get(word);
    	return Math.log(zeroSafeguard(counter.counter_spam / getSpamWordCount()));
    }
    
    private static double zeroSafeguard(double d)
    { /// zero probabilities needed to be prevented, this helper function is used in the methods above for that.
    	if (d > 0)
    		return d;
    	else
    		return EPSILON/(getNormalWordCount()+getSpamWordCount());
    }
    
/// ----------------------- Section 3 ------------------------
    
/// ---------------- 3.2 ---------------------
    
    private static void testMessages() throws IOException
    {
    	/// Start with regular messages
    	File[] messages = listing_regular;
    	MessageType outcome;
    	
    	int normalCorrect = 0, spamCorrect = 0;
    	int normalTotal, spamTotal;
    	int idx;
    	
    	for (idx = 0; idx < messages.length; ++idx)
    	{
    		outcome = classifyMsg(messages[idx]);
    		System.out.println("Message Regular #" + (idx+1) + ": " + outcome);
    		if (outcome == MessageType.NORMAL)
    			++normalCorrect;
    	}
    	normalTotal = idx;
    	
    	/// Switch to spam meessages
    	messages = listing_spam;
    	
    	for (idx = 0; idx < messages.length; ++idx)
    	{
    		outcome = classifyMsg(messages[idx]);
    		System.out.println("Message Spam #" + (idx+1) + ": " + classifyMsg(messages[idx]));
    		if (outcome == MessageType.SPAM)
    			++spamCorrect;
    	}
    	spamTotal = idx;
    	
    	DecimalFormat format = new DecimalFormat("#.##");
    	
    	System.out.println("Ratio Normal correct: " + format.format((float) normalCorrect/normalTotal) + 
    			"\nRatio Spam correct: " + format.format((float) spamCorrect/spamTotal));
    	System.out.println("Confusion Matrix:\t | Predicted\n"
    			+ "\t\t| Normal | Spam\n"
    			+ "Actual | Normal | "+ normalCorrect +"\t | "+ (normalTotal-normalCorrect) +"\n"
    			+ "       | Spam\t| "+ (spamTotal-spamCorrect) +"\t | "+ spamCorrect +"\n");
    }
    
/// ---------------- Section 4 ------------------------
    
    private static void pruneVocab() /// Now that all bigrams have been counted, remove those that don't occur enough
    {
    	Enumeration<String> keys = vocab.keys();
    	while (keys.hasMoreElements())
    	{
    		String k = keys.nextElement();
    		Multiple_Counter c = vocab.get(k);
    		if (c.counter_regular + c.counter_spam < MIN_BIGRAM_OCCURANCE)
    			vocab.remove(k);
    	}
    }
    
    private static MessageType classifyMsg(File f) throws IOException
    {
    	FileInputStream i_s = new FileInputStream( f );
        BufferedReader in = new BufferedReader(new InputStreamReader(i_s));
        String line;
        String old = ""; /// old represents the previous token, is initialised to the first
        
        double pNormal = getPriorNormal();
        double pSpam = getPriorSpam();
        
        while ((line = in.readLine()) != null)                      // read a line
        {
            StringTokenizer st = new StringTokenizer(line);         // parse it into words
            
            if (st.hasMoreTokens())
            	old = toAlpha(st.nextToken());	/// old represents the previous token, is initialised to the first

            while (st.hasMoreTokens())                  // while there are still words left..
            {
            	if (old == "")
            		old = toAlpha(st.nextToken());
            	
				String next = toAlpha(st.nextToken());
				String cat = old + " " + next;			///the concatenated strings.
				if (cat.length() >= MIN_BIGRAM_LENGTH && vocab.containsKey(cat))	/// and both words have 4 or more characters
				{														/// The -1 corrects for the space.
					pNormal += getNormalLikelihood(cat);
					pSpam += getSpamLikelihood(cat);
				}										
				old = next;
			}
        }
        
        in.close();
        
        if (pNormal > pSpam)
        	return MessageType.NORMAL;
        else
        	return MessageType.SPAM;
    }
/// ---------------- MAIN ---------------------------
    
    public static void main(String[] args)
    throws IOException /// Just throw all your exceptions upward, brilliant. If our OOP teacher could see us now...
    {
        // Location of the directory (the path) taken from the cmd line (first arg)
        File dir_location = new File( args[0] );
        
        // Check if the cmd line arg is a directory
        if ( !dir_location.isDirectory() )
        {
            System.out.println( "- Error: cmd line arg not a directory.\n" );
            Runtime.getRuntime().exit(0);
        }

        // Initialize the regular and spam lists
        listDirs(dir_location);

        // Read the e-mail messages
        readMessages(MessageType.NORMAL);
        readMessages(MessageType.SPAM);
        
        ///Cut out bigrams that do not occur enough
        pruneVocab();

        // Print out the hash table
        printVocab();
        System.out.println("Total messages | Normal: " + normalCnt + " Spam: " + spamCnt);
        System.out.println("Total words | Normal: " + normalWordCnt + " Spam: " + spamWordCnt);
        
        
        ///reset the directory to the test set.
        dir_location = new File( args[1] );
        
        if ( !dir_location.isDirectory() )
        {
            System.err.println( "- Error: second cmd line arg not a directory.\n" );
            Runtime.getRuntime().exit(0);
        }
        
        listDirs(dir_location);
        
        /// From now on listing_regular and "_spam refer to the test set!
        
        testMessages();
        
        // Now all students must continue from here:
        //
        // 1) A priori class probabilities must be computed from the number of regular and spam messages
        // 2) The vocabulary must be clean: punctuation and digits must be removed, case insensitive
        // 3) Conditional probabilities must be computed for every word
        // 4) A priori probabilities must be computed for every word
        // 5) Zero probabilities must be replaced by a small estimated value
        // 6) Bayes rule must be applied on new messages, followed by argmax classification
        // 7) Errors must be computed on the test set (FAR = false accept rate (misses), FRR = false reject rate (false alarms))
        // 8) Improve the code and the performance (speed, accuracy)
        //
        // Use the same steps to create a class BigramBayespam which implements a classifier using a vocabulary consisting of bigrams
    }
}
