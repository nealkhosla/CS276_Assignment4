package cs276.pa4;

import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;

/**
 * An abstract class for a scorer. Need to be extended by each specific implementation of scorers.
 */
public abstract class AScorer {
	
	public Map<String,Double> idfs; // Map: term -> idf
	public double totalDocuments = 0;
    // Various types of term frequencies that you will need
	String[] TFTYPES = {"url","title","body","header","anchor"};
	
	public AScorer(Map<String,Double> idfs) {
		this.idfs = idfs;
	}
	
	// Score each document for each query.
	public abstract double getSimScore(Document d, Query q);
	
	// Combines all of a documents fields for easier net computation
	public Map<String, Double> formatDocument(List<String> terms, Map<String, Map<String, Double>> tfs){
		HashMap<String,Double> combinedVector = new HashMap<String,Double>();
		for(String word: terms){
			double score = 0;
			for(String type: TFTYPES){
				Map<String,Double> sig = tfs.get(type);
				if(sig.containsKey(word)){
					score = score + sig.get(word);
				}
			}
			combinedVector.put(word,new Double(score));
		}
		return combinedVector;
	}
	
	// Handle the query vector
	public Map<String,Double> getQueryFreqs(Query q) {
		Map<String,Double> tfQuery = new HashMap<String, Double>(); // queryWord -> term frequency
		for(String word: q.words){
			if(tfQuery.containsKey(word)){
				tfQuery.put(word,new Double(tfQuery.get(word)+1));
			}else{
				tfQuery.put(word,new Double(1));
			}
		}
		return tfQuery;
	}
	
	//Does idf weighting
	public void queryIdfWeighting(Map<String,Double> tfQuery){
		for(Map.Entry<String,Double> qEntry : tfQuery.entrySet()){
			double idfScore = getTermIDF(qEntry.getKey());
			double docValue = qEntry.getValue();
			qEntry.setValue(new Double(docValue*idfScore));
		}
	}
	
	public double getTermIDF(String term){
		double idfScore = 0;
		if(!idfs.containsKey(term)){
			idfScore = Math.log10(totalDocuments+1);
		}else{
			idfScore = idfs.get(term);
		}
		return idfScore;
	}
	
	//Add word to Map
	private void addKeyNIncrement(String key, Map<String,Double> tf, int inc){
		if(tf.containsKey(key)){
			tf.put(key, new Double(tf.get(key) + inc));
		}else{
			tf.put(key, new Double(inc));
		}
	}
	//Given a url string and a query word, tokenize on non alphaneumeric characters
	//and count the times the query word appears
	private void urlTF(String qWord, String[] url, Map<String,Double> tf){
		for(String domain: url){
			if(qWord.equals(domain)){
				addKeyNIncrement(domain, tf, 1);
			}
		}
	}
	//Given a title string, tokenize on whitespace
	//and count the times the query word appears
	private void titleTF(String qWord, String[] title, Map<String,Double> tf){
		if(title == null) return;
		for(String h: title){
			if(qWord.equals(h)){
				addKeyNIncrement(h, tf, 1);
			}
		}
		
	}
	//Loop through the headers list, tokenize the strings on whitespace
	//and count the times the query word appears
	private void headerTF(String qWord, List<String> headers, Map<String,Double> tf){
		if(headers == null) return;
		for(String head: headers){
			String[] splits = head.split("\\s+");
			for(String split: splits){
				if(qWord.equals(split)){
					addKeyNIncrement(split, tf, 1);
				}
			}
		}
	}
	//Loop through the body_hits key set and if the key is equal to a query word,
	//get the size of that keys list
	private void body_hitsTF(String qWord, Map<String,List<Integer>> body_hits, Map<String,Double> tf){
		if(body_hits == null) return;
		for (Map.Entry<String, List<Integer>> entry : body_hits.entrySet()){
			String key = entry.getKey();
			if(qWord.equals(key)){
				//tf.put(key, new Double(entry.getValue().size()));
				addKeyNIncrement(key, tf, entry.getValue().size());
			}
		}
	}
	//Loop the anchors key_set and tokenize the key. If the key is equal to the query word,
	//add that keys value.
	private void anchorsTF(String qWord, Map<String, Integer> anchors, Map<String,Double> tf){
		if(anchors == null) return;
		for (Map.Entry<String, Integer> entry : anchors.entrySet()){
			String key = entry.getKey();
			String[] links = key.split("\\s+");
			HashSet<String> mySet = new HashSet<String>(Arrays.asList(links)); 
			if(mySet.contains(qWord)){
				addKeyNIncrement(qWord, tf, entry.getValue());
			}
		}
	}
	
	/*/
	 * Creates the various kinds of term frequencies (url, title, body, header, and anchor)
	 * You can override this if you'd like, but it's likely that your concrete classes will share this implementation.
	 */
	public Map<String,Map<String, Double>> getDocTermFreqs(Document d, Query q) {
		// Map from tf type -> queryWord -> score
		Map<String,Map<String, Double>> tfs = new HashMap<String,Map<String, Double>>();
		Map<String,Double> url = new HashMap<String,Double>();
		Map<String,Double> title = new HashMap<String,Double>();
		Map<String,Double> body = new HashMap<String,Double>();
		Map<String,Double> header = new HashMap<String,Double>();
		Map<String,Double> anchor = new HashMap<String,Double>();
		tfs.put(TFTYPES[0], url);
		tfs.put(TFTYPES[1], title);
		tfs.put(TFTYPES[2], body);
		tfs.put(TFTYPES[3], header);
		tfs.put(TFTYPES[4], anchor);
		// Loop through query terms and increase relevant tfs. Note: you should do this to each type of term frequencies.
		HashSet<String> hs = new HashSet<String>(q.words);
		String[] urlSplits = (d.url.toLowerCase()).split("\\W+");
		String[] titleSplits = null;
		if(d.title != null) titleSplits = (d.title).split("\\s+");
		for (String queryWord: hs) {
			urlTF(queryWord,urlSplits,url);
			titleTF(queryWord,titleSplits,title);
			headerTF(queryWord,d.headers,header);
			body_hitsTF(queryWord,d.body_hits,body);
			anchorsTF(queryWord,d.anchors,anchor);
		}
		return tfs;
	}

}
