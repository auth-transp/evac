""""""""
class Node():
    """A node class for A* Pathfinding"""

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position

"""Ορίζει την κλάση Node, που αντιπροσωπεύει έναν κόμβο (κελί) του πλέγματος.
parent: δείχνει ποιος κόμβος οδήγησε σε αυτόν (χρησιμοποιείται για την ανακατασκευή της διαδρομής).
position: συντεταγμένες (x, y) του κόμβου.
g: κόστος από την αρχή μέχρι εδώ.
h: ευρετική εκτίμηση από εδώ ως το τέλος (συνήθως τετραγωνική ή Manhattan απόσταση).
f = g + h: συνολικό εκτιμώμενο κόστος.
Η μέθοδος __eq__ επιτρέπει τη σύγκριση κόμβων μέσω θέσης (χρησιμοποιείται στις λίστες)."""


def astar(maze, start, end):
    """Κεντρική συνάρτηση που υλοποιεί τον αλγόριθμο A*. Επιστρέφει τη διαδρομή ως λίστα από συντεταγμένες"""

    # Create start and end node
    start_node = Node(None, start)
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, end)
    end_node.g = end_node.h = end_node.f = 0
    """Δημιουργεί δύο κόμβους: τον αρχικό και τον τελικό.
    Και οι δύο ξεκινούν με μηδενικά κόστη."""

    # Initialize both open and closed list
    open_list = []
    closed_list = []

    # Add the start node
    open_list.append(start_node)
    """Δύο λίστες:
    open_list: κόμβοι προς εξέταση.
    closed_list: κόμβοι που έχουν ήδη εξεταστεί.
    Ξεκινά με τον αρχικό κόμβο στην open_list."""

    # Loop until you find the end
    while len(open_list) > 0:
        """Επαναληπτικός βρόχος μέχρι να αδειάσει η open list ή να βρεθεί ο στόχος."""
    
        # Get the current node
        current_node = open_list[0]
        current_index = 0
        for index, item in enumerate(open_list):
            if item.f < current_node.f:
                current_node = item
                current_index = index
        """
        Επιλογή κόμβου με το μικρότερο f (πιο υποσχόμενος).
        Γίνεται πλήρης αναζήτηση στη λίστα (όχι βέλτιστη μέθοδος, αλλά λειτουργική).
        """
        
        # Pop current off open list, add to closed list
        open_list.pop(current_index)
        closed_list.append(current_node)
        """
        Μεταφέρει τον επιλεγμένο κόμβο από την open στη closed list.
        """

        # Found the goal
        if current_node == end_node:
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1] # Return reversed path
        """
        Αν ο τρέχων κόμβος είναι ο στόχος,
        ανακατασκευάζει τη διαδρομή προς τα πίσω μέσω parent
        και την αντιστρέφει για να ξεκινά από την αρχή.
        """

        # Generate children
        children = []
        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]: # Adjacent squares
            """
            Δημιουργεί έως 8 γειτονικούς κόμβους (ορθογώνια + διαγώνια κίνηση).
            """
            # Get node position
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])
            """
            Υπολογίζει τις συντεταγμένες κάθε παιδιού.
            """
            # Make sure within range
            if node_position[0] > (len(maze) - 1) or node_position[0] < 0 or node_position[1] > (len(maze[len(maze)-1]) -1) or node_position[1] < 0:
                continue
            """
            Απορρίπτει κόμβους εκτός ορίων πλέγματος.
            """

            # Make sure walkable terrain
            if maze[node_position[0]][node_position[1]] != 0:
                continue
            """
            Απορρίπτει εμπόδια (όπου η τιμή του κελιού ≠ 0).
            """
            
            # Create new node
            new_node = Node(current_node, node_position)

            # Append
            children.append(new_node)
        """
        Δημιουργεί έγκυρο κόμβο και τον προσθέτει στα παιδιά.
        """

        # Loop through children
        for child in children:

            # Child is on the closed list
            for closed_child in closed_list:
                if child == closed_child:
                    continue
            """
            Αγνοεί παιδιά που έχουν ήδη εξεταστεί.
            """

            # Create the f, g, and h values
            child.g = current_node.g + 1
            child.h = ((child.position[0] - end_node.position[0]) ** 2) + ((child.position[1] - end_node.position[1]) ** 2)
            child.f = child.g + child.h
            """
            Υπολογισμός κόστους:
            g: απόσταση από την αρχή (κάθε βήμα = 1).
            h: ευρετική = τετραγωνική απόσταση μέχρι το τέλος.
            f: άθροισμα.
            """

            # Child is already in the open list
            for open_node in open_list:
                if child == open_node and child.g > open_node.g:
                    continue

            # Add the child to the open list
            open_list.append(child)
            """
            Αποφεύγει προσθήκη κόμβου που υπάρχει ήδη στην open list με μικρότερο κόστος.
            Αλλιώς τον προσθέτει για μελλοντική εξέταση.
            """


def main():

    maze = [[0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    start = (0, 0)
    end = (7, 6)

    path = astar(maze, start, end)
    print(path)


if __name__ == '__main__':
    main()