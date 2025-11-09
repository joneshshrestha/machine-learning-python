import numpy as np
from numpy import linalg as la

np.set_printoptions(legacy='1.25')

############################################################################

# The following are three simialrity functions based on Euclidean distance. 
# Pearson correlation, and Cosine similarity. They have all been modified
# to produce a value between 0 and 1, with 1 representing max similarity.
# Note: These functions assume that the input vectors are Numpy arrays. 

def euclidSim(inA,inB):
    return 1.0/(1.0 + la.norm(inA - inB))

def pearsonSim(inA,inB):
    if len(inA) < 3 : return 1.0
    return 0.5+0.5*np.corrcoef(inA, inB, rowvar = 0)[0][1]

def cosineSim(inA,inB):
    num = np.dot(inA, inB)
    num = num.astype(float)
    denom = la.norm(inA)*la.norm(inB)
    return 0.5 + 0.5 * (num / denom)

############################################################################

# The following are the modifed versions of prediction and recommendation 
# functions from Machine Learning in Action (Ch. 14)

def standEst(dataMat, user, simMeas, item):
    # dataMat is assumed to be 2d Numpy array, e.g., representing a user-item rating matrix
    # user is the index of a single user (a row) in the dataMat
    # item is the index of a single item (a colums) in the dataMat
    
    n = np.shape(dataMat)[1]
    simTotal = 0.0; ratSimTotal = 0.0
    for j in range(n):
        userRating = dataMat[user,j]
        if userRating == 0: 
            continue
        overLap = np.nonzero(np.logical_and(dataMat[:,item]>0, dataMat[:,j]>0))[0]
        if len(overLap) == 0: 
            similarity = 0
        else: 
            similarity = simMeas(dataMat[overLap,item], dataMat[overLap,j])
        #print('the %d and %d similarity is: %f' % (item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0: return 0
    else: return ratSimTotal/simTotal
        
def svdEst(dataMat, user, simMeas, item):
    # dataMat is assumed to be 2d Numpy array, e.g., representing a user-item rating matrix
    # user is the index of a single user (a row) in the dataMat
    # item is the index of a single item (a colums) in the dataMat

    n = np.shape(dataMat)[1]
    simTotal = 0.0; ratSimTotal = 0.0
    # The SVD computation below requires the data to be of Numpy Matrix type
    data=np.asmatrix(dataMat)
    U,Sigma,VT = la.svd(data)
    Sig4 = np.asmatrix(np.eye(4)*Sigma[:4]) #arrange Sig4 into a diagonal matrix
    xformedItems = data.T * U[:,:4] * Sig4.I  #create transformed items (* here is matrix multiplication)
    for j in range(n):
        userRating = data[user,j]
        if userRating == 0 or j==item: continue
        similarity = simMeas(xformedItems[item,:].T, xformedItems[j,:].T)
        #print('the %d and %d similarity is: %f' % (item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0: return 0
    else: return ratSimTotal/simTotal
        
def recommend(dataMat, user, N=3, simMeas=pearsonSim, estMethod=standEst):
    unratedItems = np.nonzero(dataMat[user,:]==0)[0] #find unrated items 
    if len(unratedItems) == 0: return 'you rated everything'
    itemScores = []
    for item in unratedItems:
        estimatedScore = estMethod(dataMat, user, simMeas, item)
        itemScores.append((item, estimatedScore))
    return sorted(itemScores, key=lambda jj: jj[1], reverse=True)[:N]
    
############################################################################

# The ffollowing functions are used for evalauting the rating predictions 

def cross_validate_user(dataMat, user, test_ratio, estMethod=standEst, simMeas=pearsonSim):
	dataMat = np.array(dataMat)
	number_of_items = np.shape(dataMat)[1]
	rated_items_by_user = np.array([i for i in range(number_of_items) if dataMat[user,i]>0])
	test_size = int(test_ratio * len(rated_items_by_user))
	test_indices = np.random.randint(0, len(rated_items_by_user), test_size)
	withheld_items = rated_items_by_user[test_indices]
	original_user_profile = np.copy(dataMat[user])
	dataMat[user, withheld_items] = 0 # So that the withheld test items is not used in the rating estimation below
	error_u = 0.0
	count_u = len(withheld_items)

	# Compute absolute error for user u over all test items
	for item in withheld_items:
		# Estimate rating on the withheld item
		estimatedScore = estMethod(dataMat, user, simMeas, item)
		error_u = error_u + abs(estimatedScore - original_user_profile[item])	
	
	# Now restore ratings of the withheld items to the user profile
	for item in withheld_items:
		dataMat[user, item] = original_user_profile[item]
		
	# Return sum of absolute errors and the count of test cases for this user
	# Note that these will have to be accumulated for each user to compute MAE
	return error_u, count_u
    
def test(dataMat, test_ratio, estMethod, simMeas=pearsonSim):
    # Write this function to iterate over all users and for each perform evaluation by calling
	# the above cross_validate_user function on each user. MAE will be the ratio of total error 
	# across all test cases to the total number of test cases, across all users
    total_error = 0.0
    total_count = 0
    
    for user in range(len(dataMat)):
        user_error, user_count = cross_validate_user(dataMat, user, test_ratio, estMethod, simMeas)
        total_error += user_error
        total_count += user_count
        
    MAE = total_error/total_count
    return MAE

def load_jokes(file):
    # This function reads in the text of the jokes and returns a Numpy array of jokes
    jokes = np.genfromtxt(file, delimiter=',', dtype=str)
    jokes = np.array(jokes[:,1])
    return jokes

def get_joke_text(jokes, id):
    # This function returns the text of an individual joke given the joke id
    return jokes[id]

#####################################

# Examples:

# dataMat = np.genfromtxt('modified_jester_data.csv',delimiter=',')

# MAE = test(dataMat, 0.2, svdEst, pearsonSim)
# MAE = test(dataMat, 0.2, standEst, cosineSim)

# jokes = load_jokes('jokes.csv')
# print_most_similar_jokes(dataMat, jokes, 3, 5, pearsonSim)

''' Example output for "print_most_similar_jokes":

Selected joke: 

Q. What's the difference between a man and a toilet? A. A toilet doesn't follow you around after you use it.

Top 5 Recommended jokes are :

Q: What's the difference between a Lawyer and a Plumber? A: A Plumber works to unclog the system. 
_______________
What do you call an American in the finals of the world cup? "Hey Beer Man!" 
_______________
Q. What's 200 feet long and has 4 teeth? <P>A. The front row at a Willie Nelson Concert. 
_______________
A country guy goes into a city bar that has a dress code and the maitred' demands he wear a tie. Discouraged the guy goes to his car to sulk when inspiration strikes: He's got jumper cables in the trunk! So he wrapsthem around his neck sort of like a string tie (a bulky string tie to be sure) and returns to the bar. The maitre d' is reluctant but says to the guy "Okay you're a pretty resourceful fellow you can come in... but just don't start anything"!   
_______________
What do you get when you run over a parakeet with a lawnmower? <P>Shredded tweet. 
_______________

'''

