def show_board(board, nrows, ncols):
  for row in range(nrows):
    print(board[row*ncols:(row+1)*ncols])