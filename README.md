# CS188.1x-Artificial-Intelligence (edX)

These are the assigments for the 02.2015 edX/BerkleyX course found here: [https://www.edx.org/course/artificial-intelligence-uc-berkeleyx-cs188-1x-0](https://www.edx.org/course/artificial-intelligence-uc-berkeleyx-cs188-1x-0)

### Project 1

Files edited:
* `search.py`
  * `depthFirstSearch()`
  * `breadthFirstSearch()`
  * `uniformCostSearch()` 

Useful commands:
* `python autograder.py`
* `python pacman.py`
* `python pacman.py -l tinyMaze -p SearchAgent -a fn=tinyMazeSearch`
* DFS: `python pacman.py -l tinyMaze -p SearchAgent`
* DFS: `python pacman.py -l mediumMaze -p SearchAgent`
* DFS: `python pacman.py -l bigMaze -z .5 -p SearchAgent`
* BFS: `python pacman.py -l mediumMaze -p SearchAgent -a fn=bfs`
* BFS: `python pacman.py -l bigMaze -p SearchAgent -a fn=bfs -z .5`
* BFS: `python eightpuzzle.py`
* UCS: `python pacman.py -l mediumMaze -p SearchAgent -a fn=ucs`
* UCS: `python pacman.py -l mediumDottedMaze -p StayEastSearchAgent`
* UCS: `python pacman.py -l mediumScaryMaze -p StayWestSearchAgent`
