" Auto Setup
" ==========
if empty(glob('~/.vim/autoload/plug.vim'))
    silent !curl -fLo ~/.vim/autoload/plug.vim --create-dirs
        \ https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim
    autocmd VimEnter * PlugInstall --sync | source $MYVIMRC
endif
if empty(glob('~/.vim/colors/monokai.vim'))
    silent !curl -fLo ~/.vim/colors/monokai.vim --create-dirs
        \ https://raw.githubusercontent.com/sickill/vim-monokai/master/colors/monokai.vim
endif
if empty(glob('~/.vim/vimswap'))
    silent !mkdir -p ~/.vim/vimswap
endif

filetype plugin indent on
set tabstop=4
set shiftwidth=4
set expandtab
let g:pyindent_open_paren = '&sw'
let g:pyindent_nested_paren = '&sw'

set number
set relativenumber

set directory=~/.vim/vimswap

set wrap
set linebreak

set hidden
set wildmenu
set wildmode=longest:full,full
set fileignorecase
set ignorecase
set smartcase

set laststatus=2
set confirm
set cmdheight=1
set showcmd
set ruler

set cursorline
set noerrorbells visualbell t_vb=
set mouse=a

syntax enable
colorscheme monokai

"" mappings
nmap <S-TAB> gT
nmap <TAB> gt
nmap <C-T> :tabedit 
imap <S-TAB> <C-D>
nmap Q @q

"" vim-plug
" usage: 
" :PlugInstall
" :PlugClean
" :PlugUpdate
call plug#begin()
Plug 'terryma/vim-multiple-cursors'
let g:multi_cursor_exit_from_visual_mode=0
let g:multi_cursor_exit_from_insert_mode=0
Plug 'easymotion/vim-easymotion'
let g:EasyMotion_startofline = 0
let g:EasyMotion_smartcase = 1
let g:EasyMotion_do_shade = 0
map <Leader> <Plug>(easymotion-prefix)
nmap s <Plug>(easymotion-sn)
Plug 'tomtom/tcomment_vim'
Plug 'junegunn/vim-easy-align'
xmap ga <Plug>(LiveEasyAlign)
Plug 'tpope/vim-surround'
Plug 'raimondi/delimitmate'
Plug 'vim-airline/vim-airline'
Plug 'vim-airline/vim-airline-themes'
let g:airline#extensions#tabline#enabled=1
let g:airline_theme='powerlineish'
let g:airline#extensions#tabline#show_buffers=0
call plug#end()
