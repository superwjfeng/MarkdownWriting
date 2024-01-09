## *Vim的基本操作*

### Intro

Vi是由Bill Joy于1976年在加州大学伯克利分校开发的，作为BSD UNIX操作系统的一部分

Vim (Vi iMproved) Vim是Vi的改进版本，由Bram Moolenaar于1991年发布。它在Vi的基础上增加了许多新功能和改进，成为一个更强大、更灵活的编辑器

Neovim（简称：Nvim）是一个基于 Vim 的文本编辑器的项目，旨在作为 Vim 的改进和扩展版本。它保留了 Vim 的核心特性，并引入了一些新的功能和改进，以提供更现代、可扩展、易于配置的编辑器体验

### vim命令

vim的命令是以 `:` 开头的，即需要先键入 `:`

* 保存和退出

  * `:w`：保存当前文件

  * `:wq`：保存并退出

  * `:x` 或 `:wq`：保存并退出

  * `:q!`：强制退出，不保存修改

* 文件操作

  * `:e <文件路径>`：打开指定文件

  * `:sp <文件路径>`：水平分割窗口并打开指定文件

  * `:vsp <文件路径>`：垂直分割窗口并打开指定文件

  * `:tabe <文件路径>`：在新标签页中打开指定文件

* 查找和替换

  * `:/{模式}`：向下查找匹配的文本

  * `:?{模式}`：向上查找匹配的文本

  * `:s/{目标}/{替换}/g`：替换文本

* 移动光标

  * `:set number`：显示行号

  * `:set nonumber`：隐藏行号

  * `:set hlsearch`：高亮搜索结果

  * `:set nohlsearch`：关闭搜索结果高亮

* 窗口和标签页

  * `:vsp`：垂直分割窗口

  * `:sp`：水平分割窗口

  * `:tabnew`：新建标签页

  * `:tabnext` 或 `:tabn`：切换到下一个标签页

  * `:tabprev` 或 `:tabp`：切换到上一个标签页

* 撤销和重做

  * `:undo` 或 `:u`：撤销上一步操作

  * `:redo` 或 `:red`：重做上一步撤销的操作

* 文件信息：

  * `:pwd`：显示当前工作目录

  * `:ls`：显示缓冲区列表

  * `:edit!`：重新加载当前文件，丢弃未保存的修改

* 其他命令

  * `:help {命令}`：查看帮助文档

  * `:set {选项}`：设置 Vim 选项

  * `:map {键位}`：查看映射关系

## *vim script*

Vim script（或称为VimL）是用于编写Vim编辑器的脚本语言，它允许用户自定义和扩展Vim的功能

### 基本语法

* 注释：使用双引号 `"` 来添加注释

* 变量：使用`let`关键字来定义变量

  ```vim
  let my_variable = "Hello, Vim!"
  ```

### 控制流

* 条件语句：使用`if`、`elseif`和`endif`来实现条件语句

  ```vim
  if condition
    " 条件成立时执行的命令
  elseif another_condition
    " 另一个条件成立时执行的命令
  else
    " 如果以上条件都不成立时执行的命令
  endif
  ```

* 循环结构： 使用`for`和`endfor`进行循环操作

  ```vim
  for i in range(1, 5)
    " 循环体中的命令
  endfor
  ```

### 自定义结构

* 函数定义：使用`function`、`endfunction`定义函数

  ```vim
  function! MyFunction()
    " 函数体中的命令
  endfunction
  ```

* 自定义命令：使用`command`关键字来定义自己的命令

  ```vim
  command! Hello :echo "Hello, Vim!"
  ```

* 自动命令：通过`autocmd`关键字设置在特定事件触发时执行的命令

  ```vim
  autocmd BufRead *.txt :echo "Opening a text file"
  ```

* 调用系统命令：使用`system()`函数调用系统命令

  ```vim
  let result = system('ls -l')
  ```

## *插件安装*

Vim支持插件系统，可以通过Vim脚本编写自己的插件或使用其他人开发的插件

### 手动安装

### 使用包管理器

* 安装 vim-plug，以便它在启动时自动加载

  ```cmd
  $ curl -fLo ~/.vim/autoload/plug.vim --create-dirs \
    https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim
  ```

* 修改 `.vimrc` 配置文件

  ```vim
  call plug#begin('~/.vim/plugged')
  
  " 插件列表
  Plug '作者/插件名称'
  
  call plug#end()
  ```

* 安装插件，打开 Vim 后输入下面命令，就会安装配置文件中列出的所有插件

  ```
  :PlugInstall
  ```

### 其他的 `vim-plug` 命令

* `:PlugStatus`：查看当前插件的状态，包括已安装的插件和未安装的插件
* `:PlugUpdate`：更新配置文件中已安装插件的版本
* `:PlugUpgrade`：更新 `vim-plug` 本身到最新版本
* `:PlugClean`：删除已经不在配置文件中列出的插件
* `:PlugClean!`：强制删除所有未在配置文件中列出的插件，即使它们可能被其他插件依赖
* 快照系统
  * `:PlugDiff`：查看配置文件中插件的变化，即插件的更新或删除等
  * `:PlugSnapshot {文件路径}`：生成一个插件快照，记录当前的插件状态，可以在以后通过快照还原插件
  * `:PlugRevert`：还原到最后一次生成的插件快照

## *插件推荐*

### NerdTree

* 打开和关闭 NERDTree

  * 打开 NERDTree：在命令模式下输入 `:NERDTree` 或者按照设置使用快捷键（例如，`nt`）

  * 关闭 NERDTree：在 NERDTree 窗口中按 `q` 键

* 在 NERDTree 中导航

  * 使用上下箭头键移动光标

  * 按 `o` 键或 `<Enter>` 键打开文件或进入目录

  * 按 `O` 键递归打开目录（展开子目录）

  * 使用 `P` 键返回到上一级目录

* 在 NERDTree 中操作文件

  * 在文件上按 `m` 键，然后选择相应的操作，比如移动、复制、删除等

  * 使用 `d` 键标记文件或目录，然后按 `m` 键进行批量操作

* 切换显示模式：按 `i` 键切换 NERDTree 窗口的显示模式，可以切换为详细信息、简略信息等。

* 其他操作

  * 在 NERDTree 窗口中按 `?` 键查看帮助

  * 在 NERDTree 窗口中按 `I` 键切换是否显示隐藏文件

  * 在 NERDTree 窗口中按 `C` 键切换是否显示文件的完整路径

### Taglist

### Cscope



## *.vimrc*

### Leader 键

Leader 键的作用是为用户提供一个自定义的前缀，以便创建自己的快捷键映射。例如，如果设置了 Leader 键为逗号，那就可以通过 `,w` 来执行一个特定的操作，而不是直接使用 `:w`。这样有助于组织和管理自定义快捷键，避免与 Vim 的内置命令冲突

```vim
let g:mapleader = "," 
```

### 笔者的配置文件

```
let g:mapleader = ","

"""""""""""""""""""""""""""""""""""""
"basis settingd                     "
"""""""""""""""""""""""""""""""""""""
"high-light
syntax enable
syntax on
set nocp " 禁用兼容模式 compatible mode
set ru " 启用相对行号
"support mouse move and select 支持鼠标移动和选择
set mouse=a 
"the space number of tab-key is 2 制表符宽度
set tabstop=2
"next line keep with former line 启用自动缩进，新行的缩进将与上一行相同
set autoindent
"show line in col 80
set colorcolumn=80
"设置在使用 << 和 >> 缩进命令时的缩进宽度为4个空格
set shiftwidth=2
"check file-type 启用文件类型检测，使 Vim 可以根据文件的内容自动设置相应的文件类型 
filetype on
"show the row and line of mouse 显示光标位置的行号和列号
set ruler
"undo file 启用撤销文件功能	
set undofile
set undodir=$HOME/.vim/undodir
"
set showmatch
"show line number
set number
set list
set listchars=tab:>-,trail:-
" 
set paste
set smartindent
set matchtime=5
set cindent
" high-light show this col and row
set cursorline " 高亮显示当前光标所在行
"set cursorcolumn
"show result for search when input one character
set incsearch
"ignore low-high case for search 在搜索时忽略全小写
set smartcase
" high-light search
set hlsearch
"tab convert to space 将制表符转换为空格
set expandtab
"show cmd
set showcmd
"prohibit generate swap=file 禁止生成备份文件和交换文件
set nobackup
set noswapfile
" no high-light
nmap <F2> :noh<cr>
" support 256 colors 使用256色终端
set t_Co=256
""""""""""""""""""""""""""""""""""""""
"quick-print                         "
""""""""""""""""""""""""""""""""""""""
function PRINT()
    call append(line("."), "fflush(stdout);")
    call append(line("."), "printf(\"[%s],[%d]\\n\",__FUNCTION__,__LINE__);")
endfunction
map <F8> : call PRINT() <cr>

""""""""""""""""""""""""""""""""""""""
"winmanger setting                   "
""""""""""""""""""""""""""""""""""""""
" wm :map winmanager
nmap wm :WMToggle<cr>
let g:winManagerWindowLayout='FileExplorer'

""""""""""""""""""""""""""""""""""""""
"ctags settings                "
""""""""""""""""""""""""""""""""""""""
"loading ctags-file path
"tg :map open or close taglist
nmap tg :TlistToggle<Cr>
"nmap to :TlistOpen<cr>
"tc :map close taglist
"nmap tc :TlistClose<cr>
"ctags bin patch
let Tlist_Ctags_Cmd = '/usr/bin/ctags'
let Tlist_Show_One_File = 1
let Tlist_Exit_OnlyWindow = 1
"auto open taglist 1--auto  0--manual
let Tlist_Auto_Open=0
"taglist direction 0--right 1--left
let Tlist_Use_Right_Window = 0
"""""""""""""""""""""""""""""""""""""""
"cscpoe setting                       "
"""""""""""""""""""""""""""""""""""""""
"loading cscpoe.out
if has("cscope")
	set csprg=/usr/bin/cscope
	set csto=0
	set cst
	set nocsverb
	if filereadable("cscope.out")
		cs add cscope.out
	else
		let cscope_file=findfile("cscope.out",".;")
		let cscope_pre=matchstr(cscope_file,".*/")
		if !empty(cscope_file) && filereadable(cscope_file)
			exe "cs add" cscope_file cscope_pre
		endif
	endif
	set csverb
endif
"maping quick-key
"css :find this C symbol
nmap css :cs find s <C-R>=expand("<cword>")<CR><CR>	
"csg :find this function defination
nmap csg :cs find g <C-R>=expand("<cword>")<CR><CR>	
"csc :find the function where be called
nmap csc :cs find c <C-R>=expand("<cword>")<CR><CR>	
"cst :find this string
nmap cst :cs find t <C-R>=expand("<cword>")<CR><CR>	
nmap cse :cs find e <C-R>=expand("<cword>")<CR><CR>	
"csf :find this file
nmap csf :cs find f <C-R>=expand("<cfile>")<CR><CR>	
"csi :find this file which file include
nmap csi :cs find i ^<C-R>=expand("<cfile>")<CR>$<CR>
"csd :find function call which function
nmap csd :cs find d <C-R>=expand("<cword>")<CR><CR>

"""""""""""""""""""""""""""""""""""""""""""
" nerd-tree settings                      "
"""""""""""""""""""""""""""""""""""""""""""
" auto  open nerd-tree
autocmd VimEnter * NERDTree
" nt open/close nerdtree
nmap nt :NERDTreeToggle<cr>
" the direction of nerd-tree
let NERDTreeWinPos="right"
" show bookmarks
let NERDTreeShowBookmarks=1
" auto close nerd-tree when only alive nerd-tree
autocmd bufenter * if (winnr("$") == 1 && exists("b:NERDTree") && b:NERDTree.isTabTree()) | q | endif
" set nerd-tree show pic
let g:NERDTreeDirArrowExpandable = '▸'
let g:NERDTreeDirArrowCollapsible = '▾'
" show nerd-tree line number
"let g:NERDTreeShowLineNumbers=1
" dont show hidden file
let g:NERDTreeHidden=0
"Making it prettier
let NERDTreeMinimalUI = 1
let NERDTreeDirArrows = 1
"Show hidden file
let g:NERDTreeShowHidden = 1
"""""""""""""""""""""""""""""""""""""""""
" neocomplete settings                  "
"""""""""""""""""""""""""""""""""""""""""
"Note: This option must be set in .vimrc(_vimrc).  NOT IN .gvimrc(_gvimrc)!
" Disable AutoComplPop.
let g:acp_enableAtStartup = 0
" Use neocomplete.
let g:neocomplete#enable_at_startup = 1
" Use smartcase.
let g:neocomplete#enable_smart_case = 1
" Set minimum syntax keyword length.
let g:neocomplete#sources#syntax#min_keyword_length = 3

" Define dictionary.
let g:neocomplete#sources#dictionary#dictionaries = {
    \ 'default' : '',
    \ 'vimshell' : $HOME.'/.vimshell_hist',
    \ 'scheme' : $HOME.'/.gosh_completions'
        \ }

" Define keyword.
if !exists('g:neocomplete#keyword_patterns')
    let g:neocomplete#keyword_patterns = {}
endif
let g:neocomplete#keyword_patterns['default'] = '\h\w*'

" Plugin key-mappings.
inoremap <expr><C-g>     neocomplete#undo_completion()
inoremap <expr><C-l>     neocomplete#complete_common_string()

" Recommended key-mappings.
" <CR>: close popup and save indent.
inoremap <silent> <CR> <C-r>=<SID>my_cr_function()<CR>
function! s:my_cr_function()
  return (pumvisible() ? "\<C-y>" : "" ) . "\<CR>"
  " For no inserting <CR> key.
  "return pumvisible() ? "\<C-y>" : "\<CR>"
endfunction
" <TAB>: completion.
inoremap <expr><TAB>  pumvisible() ? "\<C-n>" : "\<TAB>"
" <C-h>, <BS>: close popup and delete backword char.
inoremap <expr><C-h> neocomplete#smart_close_popup()."\<C-h>"
inoremap <expr><BS> neocomplete#smart_close_popup()."\<C-h>"
" Close popup by <Space>.
"inoremap <expr><Space> pumvisible() ? "\<C-y>" : "\<Space>"

" AutoComplPop like behavior.
"let g:neocomplete#enable_auto_select = 1

" Shell like behavior(not recommended).
"set completeopt+=longest
"let g:neocomplete#enable_auto_select = 1
"let g:neocomplete#disable_auto_complete = 1
"inoremap <expr><TAB>  pumvisible() ? "\<Down>" : "\<C-x>\<C-u>"

" Enable omni completion.
autocmd FileType css setlocal omnifunc=csscomplete#CompleteCSS
autocmd FileType html,markdown setlocal omnifunc=htmlcomplete#CompleteTags
autocmd FileType javascript setlocal omnifunc=javascriptcomplete#CompleteJS
autocmd FileType python setlocal omnifunc=pythoncomplete#Complete
autocmd FileType xml setlocal omnifunc=xmlcomplete#CompleteTags

" Enable heavy omni completion.
if !exists('g:neocomplete#sources#omni#input_patterns')
  let g:neocomplete#sources#omni#input_patterns = {}
endif
"let g:neocomplete#sources#omni#input_patterns.php = '[^. \t]->\h\w*\|\h\w*::'
"let g:neocomplete#sources#omni#input_patterns.c = '[^.[:digit:] *\t]\%(\.\|->\)'
"let g:neocomplete#sources#omni#input_patterns.cpp = '[^.[:digit:] *\t]\%(\.\|->\)\|\h\w*::'

" For perlomni.vim setting.
" https://github.com/c9s/perlomni.vim
let g:neocomplete#sources#omni#input_patterns.perl = '\h\w*->\h\w*\|\h\w*::'

""""""""""""""""""""""""""""""""""""""""""
" rainbow settings                       "
""""""""""""""""""""""""""""""""""""""""""
"auto open rainbow
let g:rainbow_active = 1
" type
let g:rainbow_conf = {
\   'guifgs': ['royalblue3', 'darkorange3', 'seagreen3', 'firebrick'],
\   'ctermfgs': ['lightblue', 'lightyellow', 'lightcyan', 'lightmagenta'],
\   'operators': '_,\|+\|-_',
\   'parentheses': ['start=/(/ end=/)/ fold', 'start=/\[/ end=/\]/ fold', 'start=/{/ end=/}/ fold'],
\   'separately': {
\       '*': {},
\       'tex': {
\           'parentheses': ['start=/(/ end=/)/', 'start=/\[/ end=/\]/'],
\       },
\       'css': 0,
\   }
\}

"""""""""""""""""""""""""""""""""""""""""""
" nerd-commenter                          "
"""""""""""""""""""""""""""""""""""""""""""
let g:NERDSpaceDelims = 1
let g:NERDDefaultAlign = 'left'
let g:NERDToggleCheckAllLines = 1

"""""""""""""""""""""""""""""""""""""""""""
" minbufferexploer                        "
"""""""""""""""""""""""""""""""""""""""""""
let g:miniBufExplMapWindowNavVim =1
let g:miniBufExplMapWindowNavArrows = 1
let g:miniBufExplMapCTabSwitchBufs = 1

"""""""""""""""""""""""""""""""""""""""""""
" theme                                   "
"""""""""""""""""""""""""""""""""""""""""""
"colorscheme molokai
"colorscheme solarized
colorscheme gruvbox
set background=dark
let g:molokai_original = 1
let g:rehash256 = 1

"""""""""""""""""""""""""""""""""""""""""""
" interestingword settings                "
"""""""""""""""""""""""""""""""""""""""""""
nnoremap <silent> <leader>k :call InterestingWords('n')<cr>
nnoremap <silent> <leader>K :call UncolorAllWords()<cr>
nnoremap <silent> n :call WordNavigation(1)<cr>
nnoremap <silent> N :call WordNavigation(0)<cr>
"let g:interestingWordsGUIColors = ['#8CCBEA', '#A4E57E', '#FFDB72', '#FF7272', '#FFB3FF', '#9999FF']
let g:interestingWordsTermColors = ['154', '121', '211', '137', '214', '222']

"""""""""""""""""""""""""""""""""""""""""""
" LeaderF                                 "
"""""""""""""""""""""""""""""""""""""""""""
" let g:Lf_ShortcutF = '<C-P>'
" let g:Lf_WindowPosition = 'popup'
" let g:Lf_PreviewInPopup = 1

"""""""""""""""""""""""""""""""""""""""""""
"CtrlP                                    "
"""""""""""""""""""""""""""""""""""""""""""
let g:ctrlp_map = '<c-p>'
let g:ctrlp_working_path_mode = 'ra'
let g:ctrlp_switch_buffer = 'et'
let g:ctrlp_match_window = 'bottom,order:btt,min:1,max:10,results:10'

"""""""""""""""""""""""""""""""""""""""""""
" vim-airline                             "
"""""""""""""""""""""""""""""""""""""""""""
language messages en_US.utf8
set encoding=utf-8
set langmenu=en_US.utf8
set helplang=en
set ambiwidth=double
let laststatus = 2
let g:airline_powerline_fonts = 1
let g:airline_theme="dark"
let g:airline#extensions#tabline#enabled = 1
let g:airline#extensions#tabline#left_sep = ' '
let g:airline#extensions#tabline#left_alt_sep = '|'
let g:airline#extensions#tabline#buffer_nr_show = 1

"""""""""""""""""""""""""""""""""""""""""""
"syntastic                                "
"""""""""""""""""""""""""""""""""""""""""""
" let g:syntastic_enable_signs = 1
" let g:syntastic_error_symbol='✗'
" let g:syntastic_warning_symbol='►'
" let g:syntastic_always_populate_loc_list = 1
" let g:syntastic_auto_loc_list = 1
" let g:syntastic_loc_list_height = 5
" let g:syntastic_check_on_open = 1
" let g:syntastic_auto_jump = 1
" let g:syntastic_check_on_wq = 0
" let g:syntastic_enable_highlighting=1
" let g:syntastic_cpp_checkers = ['gcc']
" let g:syntastic_cpp_compiler = 'gcc'
" let g:syntastic_cpp_compiler_options = '-std=c++11'
" let g:syntastic_python_checkers = ['pyflakes']
" function! <SID>LocationPrevious()                       
"   try                                                   
"     lprev                                               
"   catch /^Vim\%((\a\+)\)\=:E553/                        
"     llast                                               
"   endtry                                                
" endfunction                                             
" function! <SID>LocationNext()                           
"   try                                                   
"     lnext                                               
"   catch /^Vim\%((\a\+)\)\=:E553/                        
"     lfirst                                              
"   endtry                                                
" endfunction                                             
" nnoremap <silent> <Plug>LocationPrevious    :<C-u>exe 'call <SID>LocationPrevious()'<CR>                                        
" nnoremap <silent> <Plug>LocationNext        :<C-u>exe 'call <SID>LocationNext()'<CR>
" nmap <silent> sp    <Plug>LocationPrevious              
" nmap <silent> sn    <Plug>LocationNext
" nnoremap <silent> <Leader>ec :SyntasticToggleMode<CR>
" function! ToggleErrors()
"     let old_last_winnr = winnr('$')
"     lclose
"     if old_last_winnr == winnr('$')
"         " Nothing was closed, open syntastic error location panel
"         Errors
"     endif
" endfunction
"""""""""""""""""""""""""""""""""""""""""""
" vim-plug settings and install           "
"""""""""""""""""""""""""""""""""""""""""""
"manager plug
call plug#begin('~/.vim/plugged')
Plug 'mhinz/vim-startify'
" YouCompleteMe
"Plug 'Valloric/YouCompleteMe'
" nerd-tree
Plug 'scrooloose/nerdtree'
" code release
Plug 'scrooloose/nerdcommenter'
" rainbow
Plug 'luochen1990/rainbow'
" fzf
"Plug 'junegunn/fzf'
"Plug '/usr/local/opt/fzf'
"Plug 'junegunn/fzf', { 'dir': '~/.fzf', 'do': './install --all' }
"Plug 'junegunn/fzf.vim'
" neoconplete
" Plug 'Shougo/neocomplete'
" wimmanager
Plug 'vim-scripts/winmanager'
" ctags
Plug 'esukram/vim-taglist'
" minibufexploer
Plug 'fholgado/minibufexpl.vim'
" molokai theme
"Plug 'tomasr/molokai'
" solarized theme
"Plug 'altercation/vim-colors-solarized'
" gruvbox theme
Plug 'morhetz/gruvbox'
" 
"Plug 'vim-airline/vim-airline-themes'
" mark char
"Plug 'mbriggs/mark.vim'
" high-light interestwaords
Plug 'lfv89/vim-interestingwords'
" neocpmplete
Plug 'shougo/neocomplete.vim'

" Plug 'Yggdroot/LeaderF', { 'do': ':LeaderfInstallCExtension' }
Plug 'ctrlpvim/ctrlp.vim'
" vim-airline"
Plug 'vim-airline/vim-airline'
Plug 'vim-airline/vim-airline-themes'
"syntastic"
" Plug 'scrooloose/syntastic'
call plug#end()
```

关于gruvbox

same thing happened to me and the above suggestion made no difference. I was able to fix my situation by creating a colors directory in my .vim directory
`mkdir ~/.vim/colors`
and then copying the gruvbox colors file to that directory
`cp ~/.vim/bundle/gruvbox/colors/gruvbox.vim ~/.vim/colors/`

https://github.com/morhetz/gruvbox/issues/85