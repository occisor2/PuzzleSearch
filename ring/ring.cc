#include <iostream>
#include <vector>
#include "mpi.h"

using std::cout ;
using std::cerr ;
using std::endl ;
using std::vector ;

int main(int argc, char *argv[])
{
  int num_processors, my_id ;

  //=========================================================================
  // All MPI programs start with initialized
  MPI_Init(&argc, &argv) ;
  MPI_Comm_size(MPI_COMM_WORLD, &num_processors) ;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_id) ;

  
  //=========================================================================
  // Print message to tell we started
  cout << "There are " << num_processors
       << " processors, my processor is number "<< my_id << endl  ;

  //=========================================================================
  // Our tie breaking for ring communication assumes an even number of
  // processors.  Check to make sure this is true.
  if(num_processors%2 != 0) {
    cerr << "Assumes an even number of processors" << endl ;
    // MPI_Abort used for abnormal exits
    MPI_Abort(MPI_COMM_WORLD,-1) ;
  }

  //=========================================================================
  // Do ring based all-to-all communication.  Each processor will start
  // with a single double value, and will end with a copy of each value
  // owned by other processors.  The ring algorithm does this in p-1
  // shifting steps where the data is copied into the array as it passes

  // data to be sent
  double send_data = double(my_id) ;

  // buffer to recv data
  vector<double> recv_data(num_processors) ;

  //=========================================================================
  // Now start all-to-all broadcast
  recv_data[my_id] = send_data ; // first copy my on processors data
  // calculate left and right processor address
  int left_processor = (my_id-1+num_processors)%num_processors ;
  int right_processor = (my_id+1)%num_processors ;
  for(int i=1;i<num_processors;++i) {
    int send_loc = (my_id+i-1)%num_processors ;
    int recv_loc = (my_id+i)%num_processors ;
    // now use odd-even tie breaking to keep processors from both
    // sending to each other potentially causing deadlock
    MPI_Status stat ;
    int tag = 0 ;
    if(my_id%2 == 0) { // even processor send first
      cout << "sending from "<< my_id << " to "<< left_processor << endl ;
      MPI_Send(&recv_data[send_loc],1,MPI_DOUBLE,
	       left_processor,tag,MPI_COMM_WORLD);
      MPI_Recv(&recv_data[recv_loc],1,MPI_DOUBLE,
	       right_processor,MPI_ANY_TAG,MPI_COMM_WORLD,&stat) ;
      cout << "processor " << my_id << " receiving msg " << recv_data[recv_loc]
	   << endl ;
    } else {  // odd processor recv first
      MPI_Recv(&recv_data[recv_loc],1,MPI_DOUBLE,
	       right_processor,MPI_ANY_TAG,MPI_COMM_WORLD,&stat) ;
      cout << "processor " << my_id << " receiving msg " << recv_data[recv_loc]
	   << endl ;
      cout << "sending from "<< my_id << " to "<< left_processor << endl ;
      MPI_Send(&recv_data[send_loc],1,MPI_DOUBLE,
	       left_processor,tag,MPI_COMM_WORLD);
    } 
  }

  // Now print out the final data
  cout << "processor " << my_id << " received: " ;
  for(int i=0;i<num_processors;++i)
    cout << recv_data[i] << ' ' ;
  cout << endl ;
  /* Were done so exit */
  MPI_Finalize() ;
  return 0 ;
}
