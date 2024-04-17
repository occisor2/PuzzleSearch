#include "game.h"
#include "utilities.h"
// Standard Includes for MPI, C and OS calls
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <mpi.h>

// C++ standard I/O and library includes
#include <iostream>
#include <fstream>
#include <tuple>
#include <utility>
#include <vector>
#include <string>

// C++ stadard library using statements
using std::cout;
using std::cerr;
using std::endl;
using std::vector;
using std::string;
using std::ofstream;
using std::ifstream;
using std::ios;

// Useful tags for distinguishing message types
enum
{
	TAG_NULL,
	TAG_REQUEST,
	TAG_RESULT,
	TAG_SOL
};
// Amount of tasks sent and recieved in a chunk
constexpr int chunkSize = 1;
// This expression is used for the size of most of the communications,
// so give it a useful name.
constexpr size_t puzzleSize = IDIM * JDIM;
// Struct for describing a result
struct Result
{
	bool found;
	int size;
	move moves[puzzleSize];
};
// Shorthand for game array
using game_t = unsigned char[puzzleSize];
// Chunk structure
struct Chunk
{
	int size;
	game_t tasks[chunkSize];
};
// Shorthand for taskbag
using TaskBag = std::vector<Chunk>;
// MPI custom data type declarations
MPI_Datatype mpi_result;
MPI_Datatype mpi_move;
MPI_Datatype mpi_game;
MPI_Datatype mpi_chunk;

/**
 * Initializes custom mpi types
 */
void createMPITypes()
{
	// mpi_move
	int m_nitems = 4;
	MPI_Datatype m_types[4] = {MPI_INT, MPI_INT, MPI_INT, MPI_UB};
	int m_blocklengths[4] = {1, 1, 1, 1};
	MPI_Aint m_offsets[4] = {offsetof(move, i), offsetof(move, j), offsetof(move, dir), sizeof(move)};

	MPI_Type_create_struct(m_nitems, m_blocklengths, m_offsets, m_types, &mpi_move);
	MPI_Type_commit(&mpi_move);

	// mpi_result
	int r_nitems = 4;
	MPI_Datatype r_types[4] = {MPI_CXX_BOOL, MPI_INT, mpi_move, MPI_UB};
	int r_blocklengths[4] = {1, 1, puzzleSize, 1};
	MPI_Aint r_offsets[4] = {offsetof(Result, found), offsetof(Result, size), offsetof(Result, moves), sizeof(Result)};

	MPI_Type_create_struct(r_nitems, r_blocklengths, r_offsets, r_types, &mpi_result);
	MPI_Type_commit(&mpi_result);
	
	// mpi_game
	MPI_Type_contiguous(puzzleSize, MPI_BYTE, &mpi_game);
	MPI_Type_commit(&mpi_game);

	// mpi_chunk
	int c_nitems = 2;
	MPI_Datatype c_types[2] = {MPI_INT, mpi_game};
	int c_blocklengths[2] = {1, chunkSize};
	MPI_Aint c_offsets[2] = {offsetof(Chunk, size), offsetof(Chunk, tasks)};

	MPI_Type_create_struct(c_nitems, c_blocklengths, c_offsets, c_types, &mpi_chunk);
	MPI_Type_commit(&mpi_chunk);
}

/**
 * Frees custom mpi types
 */
void freeMPITypes()
{
	MPI_Type_free(&mpi_move);
	MPI_Type_free(&mpi_game);
	MPI_Type_free(&mpi_result);
	MPI_Type_free(&mpi_chunk);
}

/**
 * void readGame(ifstream& is, unsigned char buff[])
 *
 * Reads a new game from the data file, is, into the buffer, buff.
 */
void readGame(ifstream& is, unsigned char buff[])
{
	string input_string;
	is >> input_string;

	if(input_string.size() != puzzleSize) {
		cerr << "something wrong in input file format!" << endl;
		MPI_Abort(MPI_COMM_WORLD, -1);
	}
	// Read in the initial game state from file
	for(int j = 0; j < puzzleSize; ++j)
	  buff[j] = input_string.at(j);
}

/**
 * void readChunk(ifstream& is, Chunk& chunk, int& gamesLeft)
 *
 * Reads as many games as there are into the Chunk object, chunk. Sets
 * the size of the chunk to how many games were actually read if there
 * aren't enough games to read.
 */
void readChunk(ifstream& is, Chunk& chunk, int& gamesLeft)
{
	int gamesRead = 0;
	
	for (auto i = 0; i < chunkSize; ++i)
	{
		// If run out of games, break early
		if (gamesLeft <= 0)
			break;
		// Read in a single game
		readGame(is, chunk.tasks[i]);
		++gamesRead;
		--gamesLeft;
	}
	
	chunk.size = gamesRead;
}

/**
 * void logSolution(ofstream& os, move solution[puzzleSize], int size, int sender, unsigned char task[])
 *
 * Logs a solution in the output file.
 */
void logSolution(ofstream& os, move solution[puzzleSize], int size, int sender, unsigned char task[])
{
	os << "found solution = " << endl;
		
	game_state s;
	s.Init(task);
	s.Print(os);
	for(int i = 0; i < size; ++i)
	{
		s.makeMove(solution[i]);
		os << "-->" << endl; 
		s.Print(os);
	}
	
	os << "solved" << endl;	
}

/**
 * std::pair<int, int> receiveResults(ofstream& os, TaskBag& taskBag)
 *
 * Waits for a result, then logs any solutions in the results if there
 * are any.
 *
 * Returns the number of results received (the same as were sent) and
 * how many of them had solutions.
 */
std::pair<int, int> receiveResults(ofstream& os, TaskBag& taskBag)
{
	int solutionsFound = 0;
	// Which client sent the result
	int clientId = 0;
	// Contains the client information
	MPI_Status status;
	// Result chunk buffer
	std::array<Result, chunkSize> results;
	
	MPI_Recv(results.data(), results.size(), mpi_result, MPI_ANY_SOURCE, TAG_RESULT, MPI_COMM_WORLD, &status);
	clientId = status.MPI_SOURCE;
	//cout << "Server: received chunk from client " << clientId<< endl;

	// Match each result to the orginal task and log the solution if
	// there was one.
	auto& chunk = taskBag.at(clientId - 1);
	//cout << "Server: logging chunk size " << chunk.size << endl;
	for (size_t i = 0; i < chunk.size; ++i)
	{	
		auto& result = results.at(i);
		if (result.found)
		{
			logSolution(os, result.moves, result.size, clientId, chunk.tasks[i]);
			++solutionsFound;
		}
	}
	
	return std::make_pair(chunk.size, solutionsFound);
}

/**
 * int handleRequest(ifstream& is, TaskBag& taskBag, int& gamesLeft)
 *
 * Wait for a request, then read a new chunk from the game file and
 * send it to whoever sends the next request. If there are no more
 * games, respond with a kill signal to the client.
 *
 * Returns the size of the chunk sent.
 */
int handleRequest(ifstream& is, TaskBag& taskBag, int& gamesLeft)
{
	// Status used to figure out who sent the request
	MPI_Status status;
	// Sender id
	int clientId = 0;
	// Wait for a request
	MPI_Recv(nullptr, 0, MPI_INT, MPI_ANY_SOURCE, TAG_REQUEST, MPI_COMM_WORLD, &status);
	clientId = status.MPI_SOURCE;
	//cout << "Server: received request from client " << clientId << endl;
	// Read a new chunk
	Chunk newChunk;
	readChunk(is, newChunk, gamesLeft);
	// Add sent tasks to the taskBag for later
	taskBag.at(clientId - 1) = newChunk;
	// Send chunk
	MPI_Send(&newChunk, 1, mpi_chunk, clientId, TAG_NULL, MPI_COMM_WORLD);
	//cout << "Server: sent chunk of size " << newChunk.size << " to client " << clientId << endl;

	return newChunk.size;
}

void Server(int argc, char *argv[]) {	
	// Check to make sure the server can run
	if(argc != 3) {
		cerr << "two arguments please!" << endl ;
		MPI_Abort(MPI_COMM_WORLD,-1) ;
	}

	cout << "Server: chunk size is " << chunkSize << endl;

	// Get the number of processes
	int processes = 0;
	MPI_Comm_size(MPI_COMM_WORLD, &processes);
	// How many puzzles had solutions
	int count = 0;
	// Input case filename 
	ifstream input(argv[1], ios::in);
	// Output case filename
	ofstream output(argv[2], ios::out);
	//output.basic_ios<char>::rdbuf(std::cout.rdbuf());
	int clientsAlive = processes - 1;
	// pending results
	int pendingResults = 0;
	// Keeps track of which processes have which puzzles;
	TaskBag taskBag(processes - 1);
	// Number of games in file
	int NUM_GAMES = 0;
	// get the number of games from the input file
	input >> NUM_GAMES;

	// Seed clients with tasks
	//cout << "Server: seeding clients" << endl;
	// Loop for as many clients as there are.
	for (int clientId = 1; clientId < processes; ++clientId)
	{
		 auto sentSize = handleRequest(input, taskBag, NUM_GAMES);
		 pendingResults += sentSize;
		 if (sentSize < chunkSize)
			 --clientsAlive;
	}

	//cout << "Server: success seeding clients" << endl;
	
	// Main server loop
	while (NUM_GAMES > 0)
	{
		// Wait for a result chunk from any client
		//cout << "Server: waiting for results" << endl;
		auto [recevied, solutions] = receiveResults(output, taskBag);
		count += solutions;
		pendingResults -= recevied;
		
		// Wait for a request and then repond appriately. If the
		// response included a kill signal, subtract that client from
		// the waiters, since it doesn't need to receive anymore.
		//cout << "Server: waiting for requests from  " << clientsAlive << endl;
		auto sentSize = handleRequest(input, taskBag, NUM_GAMES);
		pendingResults += sentSize;		
		if (sentSize < chunkSize)
			--clientsAlive;
	}
	
	//std::cout << "Server: OUT OF GAMES" << endl;

	// Receive final results
	while (pendingResults > 0)
	{
		// Wait for a result chunk from any client
		//cout << "Server: waiting for results" << endl;
		auto [recevied, solutions] = receiveResults(output, taskBag);
		count += solutions;
		pendingResults -=  recevied;
	}

	// Cleanup any lingering clients
	while (clientsAlive > 0)
	{
		handleRequest(input, taskBag, NUM_GAMES);
		--clientsAlive;
	}
	
	// Report how cases had a solution.
	std::cout << "found " << count << " solutions" << endl;
}

void Client(int id)
{
	// Keep track of how many tasks were recieved to test load
	// balancing.
	int tasksReceived = 0;
	// Loop until the server has no more tasks to give
	bool finished = false;
	
	while (not finished)
	{
		// Send request for new tasks
		//cout << "Client " << id << ": sending request" << endl;
		MPI_Send(nullptr, 0, MPI_INT, 0, TAG_REQUEST, MPI_COMM_WORLD);
		// Receive chunk
		Chunk chunk;
		MPI_Status status; // has the tag type received
		MPI_Recv(&chunk, 1, mpi_chunk, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		// Check if the chunk contained was a kill signal (size <
		// chunkSize). If the size is 0, immediately exit because the
		// server isn't expecting a response.
		if (0 == chunk.size)
			break;
		finished = chunk.size < chunkSize;
		
		// Solve chunk
		std::array<Result, chunkSize> results;
		for (size_t i = 0; i < chunk.size; ++i)
		{
			++tasksReceived;
			// Get the location of next place to place result
			auto& result = results.at(i);
			// Intialize size
			result.size = 0;
			// Initialize the game
			game_state game_board;
			game_board.Init(chunk.tasks[i]);
			// Search for a solution to the puzzle
			bool found = depthFirstSearch(game_board, result.size, result.moves);
			// Set whether a solution was found in the result
			result.found = found;
		}

		// Send results chunk
		MPI_Send(results.data(), results.size(), mpi_result, 0, TAG_RESULT, MPI_COMM_WORLD);
	}
	// Log how many tasks were received
	//cout << "Client " << id << " exited: received " << tasksReceived << " tasks." << endl;
}


int main(int argc, char* argv[])
{
	// This is a utility routine that installs an alarm to kill off this
	// process if it runs to long.  This will prevent jobs from hanging
	// on the queue keeping others from getting their work done.
	chopsigs_();
  
	// All MPI programs must call this function
	MPI_Init(&argc,&argv);
  
	int myId;
	int numProcessors;

	/* Get the number of processors and my processor identification */
	MPI_Comm_size(MPI_COMM_WORLD, &numProcessors);
	MPI_Comm_rank(MPI_COMM_WORLD, &myId);
	// Intialize custom types
	createMPITypes();

	if(myId == 0) {
		// Processor 0 runs the server code
		get_timer(); // zero the timer
		Server(argc,argv);
		// Measure the running time of the server
		cout << "Server: execution time = " << get_timer() << " seconds." << endl;
	} else {
		// All other processors run the client code.
		Client(myId);
	}

	// Deallocate custom types
	freeMPITypes();
	// All MPI programs must call this before exiting
	MPI_Finalize();
}
